"""
alfworld_rl_v1.py — AlfWorld RL v1 原始版（存档复现）
问题1: 奖励在 env.step() 前计算，success 仅表示「agent 能生成动作」
问题2: 状态向量 [0.5, 0.5, step_ratio, 1.0] 含固定常数，RL 无法感知环境
问题3: epsilon=0.2 固定不衰减
问题4: TemporaryMemory 无限累积（memory_size 配置大数模拟）
"""
import numpy as np
import time
from .rl_module_v1 import AutoToolRLAgentV1

# 模块级 DQN（全局跨 episode 学习）
_global_rl_agent_v1 = AutoToolRLAgentV1(state_dim=4, action_dim=3)

import json
from agents import load_agent
from environment import load_environment
from common.registry import registry
import copy

from utils.logging.logger import TaskLogger
from utils.logging.agent_logger import AgentLogger
logger = AgentLogger(__name__)

from .base_task import BaseTask

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}


@registry.register_task("alfworld_rl_v1")
class EvalalfworldRLv1(BaseTask):
    def __init__(self, llm_config=None, agent_name='agent_name', max_num_steps=50,
                 num_exams=134, init_prompt_path='prompts/alfworld_base.json',
                 agent_config=None, env_config=None, llm=None,
                 baseline_dir=None, log_path=None):
        super().__init__()
        self.agent = load_agent(agent_name, agent_config, llm)
        with open(init_prompt_path, 'r') as f:
            self.prompts = json.load(f)
        self.env_cfg = env_config
        self.max_num_steps = max_num_steps
        self.num_exams = num_exams
        self.baseline_dir = baseline_dir
        self.log_path = log_path
        self.agentboard = TaskLogger(
            task_name="alfworld", log_path=log_path,
            max_num_steps=self.max_num_steps, baseline_dir=self.baseline_dir
        )

    def parseAction(self, action):
        action = action.strip()
        if "put" in action:
            if " in " in action:
                action = action.replace(" in ", ' in/on ')
            elif " on " in action:
                action = action.replace(" on ", ' in/on ')
        if action.endswith('.'):
            action = action[:-1].strip()
        return action

    def evaluate_env(self, index, ob='', examples=None):
        t0 = time.time()
        init_ob = ob.split('\n')[0]
        goal = ob.split('\n')[1].split("Your task is to:")[1].strip()
        self.agent.reset(goal=goal, init_obs=init_ob)
        logger.goal("Example {} | Goal: {}".format(index, self.agent.goal))
        init_prompt_dict = copy.deepcopy(self.prompts)
        init_prompt_dict['examples'] = examples

        reward = 0.
        done = False
        grounding_acc_count = 0
        score_change_record = []
        a0_count = a1_count = a2_count = 0

        logger.info("Step {:02} - Message: {}".format(0, init_ob))
        trajectory = [{"Goal": goal, "Observation": init_ob, "id": 0}]

        for i in range(self.max_num_steps):
            # ── RL state（v1 问题2：固定常数，无法感知环境）──────────────
            step_ratio = min(i / 50.0, 1.0)
            curr_state = np.array([0.5, 0.5, step_ratio, 1.0])

            rl_action = _global_rl_agent_v1.choose_action(curr_state)
            dyn_thresh = {0: 1.5, 1: 0.6, 2: 0.1}.get(rl_action, 0.2)
            if rl_action == 0: a0_count += 1
            elif rl_action == 1: a1_count += 1
            else: a2_count += 1

            if hasattr(self.agent, 'inertia_threshold'):
                self.agent.inertia_threshold = dyn_thresh

            print(f"🧠 [RL-v1] Step {i} | Action {rl_action} -> θ={dyn_thresh} | epsilon={_global_rl_agent_v1.epsilon:.3f}", flush=True)

            # ── Agent step ────────────────────────────────────────────────
            success, action = self.agent.run(init_prompt_dict=init_prompt_dict)

            # ── RL reward（v1 问题1：在 env.step() 前计算，success=能生成动作）
            r = -0.1
            if success:
                if rl_action == 2:   r = 10.0   # 激进惯性（最省钱）
                elif rl_action == 1: r = 8.0    # 保守惯性
                else:                r = 4.0    # 强制LLM（最贵）
            elif i == self.max_num_steps - 1:
                r = -1.0

            _global_rl_agent_v1.store_transition(curr_state, rl_action, r, np.zeros(4), False)
            _global_rl_agent_v1.learn()

            if not success:
                break

            action = self.parseAction(action)
            if action in self.env.get_action_space():
                grounding_acc_count += 1.0

            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action": action, "id": i})

            observation, reward, done, info = self.env.step(action)
            logger.info("Step {:02} - Observation: {}".format(i, observation))
            logger.info("Step {:02} - Progress Rate: {}\n".format(i, reward))
            trajectory.append({"Observation": observation, "Progress Rate": reward, "id": i})

            if reward > 0:
                score_change_record.append((i, reward))

            self.agent.update(action=action, state=observation)

            if done:
                game_name = self.env.cur_task_name.split('/')[0]
                env_details = {"task_name": game_name, "goal": self.agent.goal,
                               "difficulty": self.env.difficulty}
                self.agentboard.log_example(
                    index, True, reward, grounding_acc_count / (i + 1),
                    score_change_record, env_details, trajectory
                )
                elapsed = time.time() - t0
                print(f"[TIMER] Example {index}: elapsed={elapsed:.2f}s", flush=True)
                print(f"[INERTIA] Example {index}: steps={i+1}, A0={a0_count}, A1={a1_count}, A2={a2_count}", flush=True)
                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

        game_name = self.env.cur_task_name.split('/')[0]
        env_details = {"task_name": game_name, "goal": self.agent.goal,
                       "difficulty": self.env.difficulty}
        self.agentboard.log_example(
            index, done, reward, grounding_acc_count / (i + 1),
            score_change_record, env_details, trajectory
        )
        elapsed = time.time() - t0
        print(f"[TIMER] Example {index}: elapsed={elapsed:.2f}s", flush=True)
        print(f"[INERTIA] Example {index}: steps={i+1}, A0={a0_count}, A1={a1_count}, A2={a2_count}", flush=True)
        return reward, done, grounding_acc_count / (i + 1), score_change_record, i

    def evaluate(self):
        self.env = load_environment('alfworld', self.env_cfg)
        scores, srs, grounding_accs, score_state_records, difficulties = [], [], [], [], []

        for id in range(self.num_exams):
            ob, info = self.env.reset()
            ob = '\n'.join(ob[0].split('\n\n')[1:])
            name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
            difficulties.append(self.env.difficulty)

            for k, v in prefixes.items():
                if name.startswith(k):
                    score, is_done, grounding_acc, score_change_record, steps = \
                        self.evaluate_env(ob=ob, examples="".join(self.prompts['examples'][v]), index=id)
                    srs.append(1.0 if is_done else 0.0)
                    scores.append(score)
                    grounding_accs.append(grounding_acc)
                    score_state_records.append(score_change_record)
                    logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(
                        id, is_done, score, steps))

        sr = sum(srs) / len(srs)
        pr = sum(scores) / len(scores)
        gr = sum(grounding_accs) / len(grounding_accs)
        easy_srs = [s for s, d in zip(srs, difficulties) if d == "easy"]
        hard_srs = [s for s, d in zip(srs, difficulties) if d == "hard"]
        easy_prs = [s for s, d in zip(scores, difficulties) if d == "easy"]
        hard_prs = [s for s, d in zip(scores, difficulties) if d == "hard"]
        easy_sr = sum(easy_srs) / len(easy_srs) if easy_srs else 0
        hard_sr = sum(hard_srs) / len(hard_srs) if hard_srs else 0
        easy_pr = sum(easy_prs) / len(easy_prs) if easy_prs else 0
        hard_pr = sum(hard_prs) / len(hard_prs) if hard_prs else 0

        self.agentboard.log_summary(sr, pr, gr, score_state_records, hard_sr, hard_pr, easy_sr, easy_pr)
        return srs, scores, grounding_accs, score_state_records, easy_sr, hard_sr, easy_pr, hard_pr

    @classmethod
    def from_config(cls, run_config, llm_config, agent_config, env_config, llm=None):
        return cls(
            llm_config=llm_config,
            agent_name=agent_config.get("name", "ReactInertiaAgent"),
            max_num_steps=run_config.get("max_num_steps", 50),
            num_exams=run_config.get("num_exam", 134),
            init_prompt_path=agent_config.get("init_prompt_path", "prompts/alfworld_base.json"),
            agent_config=agent_config,
            env_config=env_config,
            llm=llm,
            baseline_dir=run_config.get("baseline_dir", "data/baseline_results"),
            log_path=run_config.get("log_path", None),
        )
