"""
webshop_flat.py — Flat (non-hierarchical) WebShop task.

Registers as task type "webshop_flat".
Supports both ReactBaselineAgent (ReAct) and ReactInertiaAgent (AutoTool/AdaInertia).
Uses the offline BM25 WebShop environment — no Flask server required.
"""

import json
import os
import re
import time

from agents import load_agent
from environment import load_environment
from common.registry import registry
from utils.logging.logger import TaskLogger
from utils.logging.agent_logger import AgentLogger
from .base_task import BaseTask

logger = AgentLogger(__name__)

try:
    from autool.utils.call_model import reset_token_stats, get_token_stats
except ImportError:
    def reset_token_stats(): pass
    def get_token_stats(): return {"input_tokens": 0, "output_tokens": 0, "n_calls": 0}


@registry.register_task("webshop_flat")
class EvalWebshopFlat(BaseTask):

    def __init__(self,
                 llm_name="gpt",
                 llm_config=None,
                 agent_name="ReactBaselineAgent",
                 agent_config=None,
                 env_config=None,
                 run_config=None,
                 llm=None,
                 baseline_dir=None,
                 log_path=None):

        super().__init__()
        self.agent = load_agent(agent_name, agent_config, llm)
        self.env_cfg = env_config

        self.max_num_steps  = int(run_config.get("max_num_steps", 20))
        self.num_exam       = int(run_config.get("num_exam", 251))
        self._log_path_dir  = log_path
        self.baseline_dir   = baseline_dir
        self.label_path     = env_config.get("label_path", None)

        self.agentboard = TaskLogger(
            task_name="webshop",
            log_path=log_path,
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _load_labels(self):
        if not self.label_path or not os.path.exists(self.label_path):
            raise FileNotFoundError(f"WebShop label file not found: {self.label_path}")
        labels = []
        with open(self.label_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(json.loads(line))
        return labels

    @staticmethod
    def clean_action(message):
        """Extract search[...] or click[...] from LLM output."""
        if not message:
            return message
        m = re.search(r"(click\[|search\[).*", message, re.IGNORECASE)
        return message[m.start():].split('\n')[0].strip() if m else message.strip()

    # ── Checkpoint helpers ─────────────────────────────────────────────────

    def _checkpoint_path(self):
        return os.path.join(self._log_path_dir or "", "checkpoint.jsonl")

    def _load_completed(self):
        completed = {}
        ckpt = self._checkpoint_path()
        if os.path.exists(ckpt):
            with open(ckpt) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        completed[rec["index"]] = rec
                    except Exception:
                        pass
        return completed

    def _save_checkpoint(self, index, done, score, grounding_acc,
                         score_change_record, num_steps, tok, difficulty):
        ckpt = self._checkpoint_path()
        rec = {
            "index": index, "done": done, "score": score,
            "grounding_acc": grounding_acc,
            "score_change_record": score_change_record,
            "num_steps": num_steps, "tokens": tok, "difficulty": difficulty,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(ckpt, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ── Episode evaluation ─────────────────────────────────────────────────

    def evaluate_env(self, index, session_id, goal, difficulty):
        # Initial reset
        obs, reward, done, sub_reward, grounding = self.env.step(session_id, "reset[]")
        self.agent.reset(goal=goal, init_obs=obs)

        last_sub = 0.0
        grounding_acc_count = 0
        score_change_record = []
        trajectory = [{"Goal": goal, "id": 0}, {"Observation": obs, "id": 0}]

        logger.info("Step 00 - Observation: {}".format(obs))

        for i in range(self.max_num_steps):
            success, action_raw = self.agent.run()
            action = self.clean_action(action_raw) if action_raw else ""

            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action": action, "id": i})

            if not success or not action:
                break

            obs, reward, done, sub_reward, grounding = self.env.step(session_id, action)
            if grounding:
                grounding_acc_count += 1

            logger.info("Step {:02} - Observation: {}".format(i, obs))
            trajectory.append({"Observation": obs, "id": i})
            trajectory.append({"Progress Rate": sub_reward, "id": i})

            if sub_reward > last_sub:
                score_change_record.append((i, sub_reward))
            last_sub = sub_reward

            if done:
                is_success = (reward >= 1.0)
                env_details = {"task_name": "webshop", "goal": goal, "difficulty": difficulty}
                self.agentboard.log_example(
                    index, is_success, sub_reward,
                    grounding_acc_count / (i + 1),
                    score_change_record, env_details, trajectory,
                )
                return sub_reward, is_success, grounding_acc_count / (i + 1), score_change_record, i

            self.agent.update(action=action, state=obs)

        env_details = {"task_name": "webshop", "goal": goal, "difficulty": difficulty}
        try:
            example_prompt = self.agent.get_example_prompt()
        except Exception:
            example_prompt = None
        self.agentboard.log_example(
            index, False, last_sub,
            grounding_acc_count / max(i + 1, 1),
            score_change_record, env_details, trajectory, example_prompt,
        )
        return last_sub, False, grounding_acc_count / max(i + 1, 1), score_change_record, i

    # ── Main evaluate loop ─────────────────────────────────────────────────

    def evaluate(self):
        labels = self._load_labels()[:self.num_exam]
        self.env = load_environment("webshop", self.env_cfg)
        print(f"[WebShop-Flat] Running {len(labels)} tasks", flush=True)

        completed = self._load_completed()
        if completed:
            print(f"[RESUME] Found {len(completed)} completed, skipping.")

        srs, scores, grounding_accs, score_state_records, difficulties = [], [], [], [], []

        for index, task in enumerate(labels):
            goal       = task.get("goal", "")
            difficulty = task.get("difficulty", "easy")
            session_id = f"fixed_{index}"

            if index in completed:
                c = completed[index]
                srs.append(1.0 if c["done"] else 0.0)
                scores.append(c["score"])
                grounding_accs.append(c["grounding_acc"])
                score_state_records.append(c.get("score_change_record", []))
                difficulties.append(c.get("difficulty", "easy"))
                print(f"[RESUME] Skip {index} (SR={c['done']}, PR={c['score']:.3f})")
                continue

            logger.goal(f"Example {index} | {goal}")
            reset_token_stats()
            try:
                score, done, grounding_acc, score_change_record, num_steps = \
                    self.evaluate_env(index, session_id, goal, difficulty)
            except Exception as e:
                print(f"[ERROR] Example {index}: {e}", flush=True)
                score, done, grounding_acc, score_change_record, num_steps = 0.0, False, 0.0, [], 0
            tok = get_token_stats()

            difficulties.append(difficulty)
            logger.finish(f"Example {index} | SR={done}, PR={score:.3f}, steps={num_steps+1}")
            print(f"[TOKEN] {index}: in={tok['input_tokens']}, out={tok['output_tokens']}, calls={tok['n_calls']}", flush=True)

            self._save_checkpoint(index, done, score, grounding_acc,
                                  score_change_record, num_steps + 1, tok, difficulty)

            srs.append(1.0 if done else 0.0)
            scores.append(score)
            grounding_accs.append(grounding_acc)
            score_state_records.append(score_change_record)

        sr = sum(srs) / len(srs)
        pr = sum(scores) / len(scores)
        gr = sum(grounding_accs) / len(grounding_accs)

        hard_sr = [s for s, d in zip(srs,    difficulties) if d == "hard"]
        hard_pr = [s for s, d in zip(scores, difficulties) if d == "hard"]
        easy_sr = [s for s, d in zip(srs,    difficulties) if d == "easy"]
        easy_pr = [s for s, d in zip(scores, difficulties) if d == "easy"]

        hard_sr = sum(hard_sr) / len(hard_sr) if hard_sr else 0.0
        hard_pr = sum(hard_pr) / len(hard_pr) if hard_pr else 0.0
        easy_sr = sum(easy_sr) / len(easy_sr) if easy_sr else 0.0
        easy_pr = sum(easy_pr) / len(easy_pr) if easy_pr else 0.0

        self.agentboard.log_summary(sr, pr, gr, score_state_records,
                                    hard_sr, hard_pr, easy_sr, easy_pr)
        return srs, scores, grounding_accs, score_state_records, easy_sr, hard_sr, easy_pr, hard_pr

    @classmethod
    def from_config(cls, run_config, llm_config, agent_config, env_config, llm=None):
        return cls(
            llm_name     = llm_config.get("name", "gpt"),
            llm_config   = llm_config,
            agent_name   = agent_config.get("name", "ReactBaselineAgent"),
            agent_config = agent_config,
            env_config   = env_config,
            run_config   = run_config,
            llm          = llm,
            baseline_dir = run_config.get("baseline_dir", "data/baseline_results"),
            log_path     = run_config.get("log_path", None),
        )
