"""
scienceworld_dqn.py — ScienceWorld task with DQN-based RL (RL v1 original version).
Registers as task type "scienceworld_dqn" to coexist with standard "scienceworld".

Based on backup_v2/code/scienceworld.py — the original DQN implementation.
"""
import numpy as np
import os
import json
import re
import time
import random
from agents import load_agent
from environment import load_environment
from common.registry import registry

from .rl_module_dqn import AutoToolRLAgent

# Module-level DQN agent (fresh start, no pre-trained weights)
_global_rl_agent = AutoToolRLAgent(state_dim=4, action_dim=3)

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


@registry.register_task("scienceworld_dqn")
class EvalScienceworldDQN(BaseTask):
    def __init__(self,
                 llm_name="gpt",
                 llm_config=None,
                 agent_name="GPTAgent",
                 agent_config=None,
                 env_config=None,
                 run_config=None,
                 llm=None,
                 baseline_dir=None,
                 log_path=None
                 ):

        super().__init__()

        self.agent = load_agent(agent_name, agent_config, llm)
        self.simplefied = env_config.get("simplefied", False)
        seed = env_config.get("seed", 42)
        self.set_seed(seed)
        self.simplification_str = self.build_simplification_str()
        self.env_cfg = env_config

        self.max_num_steps = int(run_config.get("max_num_steps", 30))
        self.context_length = llm_config.get("context_length")

        self.baseline_dir = baseline_dir

        self.agentboard = TaskLogger(
            task_name="scienceworld", log_path=log_path,
            max_num_steps=self.max_num_steps, baseline_dir=self.baseline_dir
        )

    def build_simplification_str(self):
        simplifications = [
            "selfWateringFlowerPots",
            "openContainers",
            "openDoors",
            "noElectricalAction",
        ]
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

        logger.info("Step {:02} - Observation: {}".format(0, init_obs))
        grounding_acc_count = 0
        score_change_record = []
        isDone = False

        trajectory = []
        trajectory.append({"Goal": modified_goal, "id": 0})
        trajectory.append({"Observation": init_obs, "id": 0})

        inertia_calls = 0  # for building state vector

        for i in range(self.max_num_steps):
            # === [RL Decision Layer] ===
            step_ratio = min(i / max(self.max_num_steps - 1, 1), 1.0)
            inertia_ratio = inertia_calls / max(i, 1) if i > 0 else 0.0
            avg_tokens = self.agent.token_counts.get("input_tokens", 0) / max(i, 1) if hasattr(self.agent, "token_counts") else 1500
            token_eff = max(0.0, 1.0 - avg_tokens / 4000.0)
            curr_state = np.array([last_reward, inertia_ratio, step_ratio, token_eff])
            rl_action = _global_rl_agent.choose_action(curr_state)

            # Action -> Threshold
            if rl_action == 0:
                dyn_thresh = 1.5   # Force LLM
            elif rl_action == 1:
                dyn_thresh = 0.6   # Conservative inertia
            else:
                dyn_thresh = 0.1   # Aggressive inertia

            if hasattr(self.agent, 'inertia_threshold'):
                self.agent.inertia_threshold = dyn_thresh
            if hasattr(self.agent, 'config') and isinstance(self.agent.config, dict):
                self.agent.config['inertia_threshold'] = dyn_thresh

            print(f"🧠 [RL] Step {i} | Action {rl_action} -> Set inertia_threshold to {dyn_thresh} | epsilon={_global_rl_agent.epsilon:.3f}", flush=True)
            # ===================================

            success, action = self.agent.run()

            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action": action, "id": i})

            if not success:
                _global_rl_agent.store_transition(curr_state, rl_action, -1.0, curr_state, True)
                _global_rl_agent.learn()
                break

            observation, reward, isDone, info = self.env.step(action)
            if action in self.env.get_action_space(abstract=False):
                grounding_acc_count += 1

            logger.info("Step {:02} - Observation: {}".format(i, observation))
            logger.info("Step {:02} - Progress Rate: {}\n".format(i, reward))

            trajectory.append({"Observation": observation, "id": i})
            trajectory.append({"Progress Rate": reward, "id": i})

            progress_delta = reward - last_reward
            if reward > last_reward:
                score_change_record.append((i, reward))

            # === [RL Learning Layer] ===
            if isDone:
                rl_r = 10.0
            elif progress_delta > 0:
                rl_r = progress_delta * 5.0
            else:
                rl_r = -0.1

            step_tokens = self.agent.token_counts.get("input_tokens", 0) if hasattr(self.agent, "token_counts") else 0
            if step_tokens > 2000:
                rl_r -= (step_tokens - 2000) / 20000.0

            if hasattr(self.agent, 'inertia_count'):
                inertia_calls = self.agent.inertia_count

            last_reward = reward
            next_inertia_ratio = inertia_calls / max(i + 1, 1)
            next_state = np.array([reward, next_inertia_ratio, step_ratio, token_eff])
            _global_rl_agent.store_transition(curr_state, rl_action, rl_r, next_state, isDone)
            _global_rl_agent.learn()
            # ===========================

            if isDone:
                env_details = {"task_name": task_name, "goal": self.agent.goal, "difficulty": self.env.difficulty}
                self.agentboard.log_example(index, True, 1.0, grounding_acc_count / (i + 1), score_change_record, env_details, trajectory)
                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

            self.agent.update(action=action, state=observation)

        env_details = {"task_name": task_name, "goal": self.agent.goal, "difficulty": self.env.difficulty}
        try:
            example_prompt = self.agent.get_example_prompt()
        except:
            example_prompt = None

        progress_rate = reward

        self.agentboard.log_example(index, isDone, progress_rate, grounding_acc_count / (i + 1), score_change_record, env_details, trajectory, example_prompt)

        return progress_rate, isDone, grounding_acc_count / (i + 1), score_change_record, i

    def evaluate(self):
        self.env = load_environment("scienceworld", self.env_cfg)
        labels = self.env.labels
        count = 0
        scores = []
        score_state_records = []
        grounding_accs = []
        srs = []
        difficulties = []

        for index, (k, v) in enumerate(labels.items()):
            task_name = v["task_name"]
            var = v["var"]
            modified_goal = v["modified_goal"]

            logger.goal("Example {} | Goal: {}".format(index, f"task_name: {task_name}, var: {var}, {modified_goal}"))
            reset_token_stats()
            score, done, grounding_acc, score_change_record, num_steps = self.evaluate_env(index, task_name, var, modified_goal)
            tok = get_token_stats()

            difficulties.append(self.env.difficulty)
            logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(index, done, score, num_steps + 1))
            print(f"[TOKEN] Example {index}: in={tok['input_tokens']}, out={tok['output_tokens']}, calls={tok['n_calls']}", flush=True)
            count += 1
            srs.append(1.0 if done else 0.0)
            scores.append(score)
            grounding_accs.append(grounding_acc)
            score_state_records.append(score_change_record)

        sr = sum(srs) / len(srs)
        pr = sum(scores) / len(scores)
        gr = sum(grounding_accs) / len(grounding_accs)

        hard_sr = [s for s, d in zip(srs, difficulties) if d == "hard"]
        hard_sr = sum(hard_sr) / len(hard_sr) if hard_sr else 0

        hard_pr = [s for s, d in zip(scores, difficulties) if d == "hard"]
        hard_pr = sum(hard_pr) / len(hard_pr) if hard_pr else 0

        easy_sr = [s for s, d in zip(srs, difficulties) if d == "easy"]
        easy_sr = sum(easy_sr) / len(easy_sr) if easy_sr else 0

        easy_pr = [s for s, d in zip(scores, difficulties) if d == "easy"]
        easy_pr = sum(easy_pr) / len(easy_pr) if easy_pr else 0

        self.agentboard.log_summary(sr, pr, gr, score_state_records, hard_sr, hard_pr, easy_sr, easy_pr)

        return srs, scores, grounding_accs, score_state_records, easy_sr, hard_sr, easy_pr, hard_pr

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

        return cls(
            llm_name=llm_name,
            llm_config=llm_config,
            agent_name=agent_name,
            agent_config=agent_config,
            env_config=env_config,
            run_config=run_config,
            llm=llm,
            baseline_dir=baseline_dir,
            log_path=log_path,
        )
