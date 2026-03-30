import os
import pdb
import json
import re
import time
from agents import load_agent
import random
from environment import load_environment
from common.registry import registry

from utils.logging.logger import TaskLogger
from utils.logging.agent_logger import AgentLogger
logger = AgentLogger(__name__)

from .base_task import BaseTask

try:
    from autool.utils.call_model import reset_token_stats, get_token_stats
    _TOKEN_TRACKING = True
except ImportError:
    _TOKEN_TRACKING = False
    def reset_token_stats(): pass
    def get_token_stats(): return {"input_tokens": 0, "output_tokens": 0, "n_calls": 0}


@registry.register_task("scienceworld")
class EvalScienceworld(BaseTask):
    def __init__(self,
                 llm_name="gpt",
                 llm_config=None,
                 agent_name="GPTAgent",
                 agent_config=None,
                 env_config=None,
                 run_config=None,
                 llm=None,
                 baseline_dir = None,
                 log_path = None
                 ):
        
        super().__init__()
        
        self.agent = load_agent(agent_name, agent_config, llm)
        self.simplefied = env_config.get("simplefied", False)
        seed = env_config.get("seed", 42)
        self.set_seed(seed)
        self.simplification_str = self.build_simplification_str()
        self.env_cfg = env_config
        
        # change the name max_episode to max_num_step for consistency
        self.max_num_steps = run_config.get("max_num_steps", 30)
        self.context_length = llm_config.get("context_length")

        self.baseline_dir = baseline_dir

        # Read existing results BEFORE TaskLogger clears the file (resume support)
        self._resume_data = {}
        _existing_txt = os.path.join(log_path, "scienceworld.txt") if log_path else None
        if _existing_txt and os.path.exists(_existing_txt):
            with open(_existing_txt) as _f:
                for _line in _f:
                    _m = re.match(r'\[EXP\] (\d+): \[success_rate\]: (\w+), \[progress_rate\]: ([\d.]+), \[grounding_acc\]: ([\d.]+)', _line)
                    if _m:
                        _idx = int(_m.group(1))
                        _sr = 1.0 if _m.group(2) == 'True' else 0.0
                        _pr = float(_m.group(3))
                        _ga = float(_m.group(4))
                        self._resume_data[_idx] = (_sr, _pr, _ga)
            if self._resume_data:
                print(f"[RESUME] Loaded {len(self._resume_data)} completed examples from {_existing_txt}", flush=True)

        self.agentboard = TaskLogger(task_name="scienceworld", log_path=log_path, max_num_steps=self.max_num_steps, baseline_dir=self.baseline_dir)

        
    def build_simplification_str(self):

        simplifications = list()
        simplifications.append("selfWateringFlowerPots")
        simplifications.append("openContainers")
        simplifications.append("openDoors")
        simplifications.append("noElectricalAction")

        return ",".join(simplifications)

    def set_seed(self, seed):
        random.seed(seed)

    def evaluate_env(self, index, task_name, var, modified_goal):
        self.env.load(task_name, var, simplificationStr=self.simplification_str)
        initialObs, initialDict = self.env.reset()
        init_obs = initialObs + f"\n{self.env.inventory()}"
        self.agent.reset(goal=modified_goal, init_obs=init_obs)
        reward = 0.
        last_reward = 0.
       # print(init_obs)
        logger.info("Step {:02} - Observation: {}".format(0, init_obs))
        grounding_acc_count = 0
        score_change_record = []
        isDone = False
        
        trajectory = []
        trajectory.append({"Goal":modified_goal, "id":0})
        trajectory.append({"Observation":init_obs, "id":0})  
        
        for i in range(self.max_num_steps):
            success, action = self.agent.run()
            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action":action, "id":i})
            
            if not success:
                break

            observation, reward, isDone, info = self.env.step(action)
            if action in self.env.get_action_space(abstract=False):
                grounding_acc_count += 1
                
            #print(f"step: {i} ACTION: {action}\nOBSERVATION: {observation}")
            logger.info("Step {:02} - Observation: {}".format(i, observation))
            logger.info("Step {:02} - Progress Rate: {}\n".format(i, reward))
            
            trajectory.append({"Observation":observation, "id":i})
            trajectory.append({"Progress Rate":reward, "id":i})
            
            if reward > last_reward:
                score_change_record.append((i, reward))
            last_reward = reward
            if isDone:
                
                env_details = {"task_name": task_name, "goal": self.agent.goal, "difficulty": self.env.difficulty}
                
                self.agentboard.log_example(index, True, 1.0, grounding_acc_count / (i + 1), score_change_record, env_details, trajectory)
               
                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i
            
            self.agent.update(action=action,
                              state=observation)

        
        env_details = {"task_name": task_name, "goal": self.agent.goal, "difficulty": self.env.difficulty}
        try: example_prompt = self.agent.get_example_prompt()
        except: example_prompt = None  
        
        progress_rate = reward
        
        self.agentboard.log_example(index, isDone, progress_rate, grounding_acc_count / (i + 1), score_change_record, env_details, trajectory, example_prompt)

        return progress_rate, isDone, grounding_acc_count / (i + 1), score_change_record, i

    def evaluate(self):
        scores = []
        self.env = load_environment("scienceworld", self.env_cfg)
        labels = self.env.labels
        count = 0
        scores = []
        score_state_records = []
        grounding_accs = []
        srs = []
        difficulties = []

        # Resume: use pre-loaded completed examples (loaded before TaskLogger cleared the file)
        completed = getattr(self, '_resume_data', {})
        if completed:
            print(f"[RESUME] Skipping {len(completed)} already-completed examples.", flush=True)

        for index, (k, v) in enumerate(labels.items()):

            
            task_name = v["task_name"]
            var = v["var"]
            modified_goal = v["modified_goal"]

            # Skip already-completed examples (resume support)
            if index in completed:
                _sr, _pr, _ga = completed[index]
                srs.append(_sr)
                scores.append(_pr)
                grounding_accs.append(_ga)
                score_state_records.append([])
                difficulties.append("hard")  # default
                # Re-write result to file (TaskLogger cleared it on init)
                _done_bool = (_sr == 1.0)
                with open(self.agentboard.log_summary_path, "a+") as _wf:
                    _wf.write(f"[EXP] {index}: [success_rate]: {_done_bool}, [progress_rate]: {_pr}, [grounding_acc]: {_ga}, [score_state]: [] \n")
                count += 1
                continue

            #print(f"Starting Task: {task_name}, variation: {var}, goal: {modified_goal}")
            logger.goal("Example {} | Goal: {}".format(index, f"task_name: {task_name}, var: {var}, {modified_goal}"))
            reset_token_stats()
            _t_start = time.time()
            score, done, grounding_acc, score_change_record, num_steps = self.evaluate_env(index, task_name, var, modified_goal)
            _t_elapsed = time.time() - _t_start
            tok = get_token_stats()

            difficulties.append(self.env.difficulty)
            logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(index, done, score, num_steps + 1))
            print(f"[TOKEN] Example {index}: in={tok['input_tokens']}, out={tok['output_tokens']}, calls={tok['n_calls']}", flush=True)
            print(f"[TIMER] Example {index}: elapsed={_t_elapsed:.2f}s", flush=True)
            # Log RL strategy distribution (A0/A1/A2) per example
            _total_steps = num_steps + 1
            _a1 = getattr(self.agent, 'inertia_count', 0)
            _a2 = getattr(self.agent, 'hint_count', 0)
            _a0 = _total_steps - _a1 - _a2
            print(f"[INERTIA] Example {index}: steps={_total_steps}, A0={_a0}, A1={_a1}, A2={_a2}", flush=True)
            count += 1
            if done:
                srs.append(1.0)
            else:
                srs.append(0.0)
            scores.append(score)
            grounding_accs.append(grounding_acc)
            score_state_records.append(score_change_record)
            #print(f"task: {task_name}, var: {var}, score: {score}, done: {done}" )

        #print(f"avg score: {sum(scores) * 1.0 / len(scores)}, SR: {sum(srs) * 1.0 / len(srs)}")

        sr = sum(srs) * 1.0 / len(srs)
        pr = sum(scores) * 1.0 / len(scores)
        gr = sum(grounding_accs) * 1.0 / len(grounding_accs)

        hard_sr = [sr for sr, difficulty in zip(srs, difficulties) if difficulty == "hard"]
        hard_sr = sum(hard_sr) / len(hard_sr) if len(hard_sr) > 0 else 0

        hard_pr = [pr for pr, difficulty in zip(scores, difficulties) if difficulty == "hard"]
        hard_pr = sum(hard_pr) / len(hard_pr) if len(hard_pr) > 0 else 0

        easy_sr = [sr for sr, difficulty in zip(srs, difficulties) if difficulty == "easy"]
        easy_sr = sum(easy_sr) / len(easy_sr) if len(easy_sr) > 0 else 0

        easy_pr = [pr for pr, difficulty in zip(scores, difficulties) if difficulty == "easy"]
        easy_pr = sum(easy_pr) / len(easy_pr) if len(easy_pr) > 0 else 0
                    
        
        self.agentboard.log_summary(sr, pr, gr, score_state_records, hard_sr, hard_pr, easy_sr, easy_pr)

        return  srs, scores, grounding_accs, score_state_records, easy_sr, hard_sr, easy_pr, hard_pr

    def _grounding_fn(self, action):
        valid_actions = self.env.GetValidActions()
        return "check valid actions" if action not in valid_actions else action

    @classmethod
    def from_config(cls,
                    run_config,
                    llm_config,
                    agent_config,
                    env_config,
                    llm=None
                    ):
        llm_name = llm_config.get("name", "gpt")
        agent_name = agent_config.get("name", "GPTAgent")
        baseline_dir = run_config.get("baseline_dir", "data/baseline_results")
        log_path = run_config.get("log_path", None)
                
        return cls(llm_name=llm_name,
                   llm_config=llm_config,
                   agent_name=agent_name,
                   agent_config=agent_config,
                   env_config=env_config,
                   run_config=run_config,
                   llm=llm,
                   baseline_dir=baseline_dir,
                   log_path = log_path
                   )


