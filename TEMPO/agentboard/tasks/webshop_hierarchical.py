"""
webshop_hierarchical.py — WebShop task with HierInertia (Semi-MDP).

Registers as task type "webshop_hierarchical".

Architecture mirrors scienceworld_hierarchical.py but adapts to the
offline WebShop environment:
  - Offline BM25 product search (no Flask server needed)
  - Session-based navigation: search → product → buy
  - Reward: price_match × option_score × attribute_score (0.0–1.0)
  - Success: score == 1.0 AND click[Buy Now] issued

Phase detector behavior on WebShop (dense partial reward):
  - EXPLORATION: Initial search phase, high threshold (cautious)
  - EXECUTION:   Narrowing in on product, lower threshold (aggressive)
  - TRANSITION:  Switching strategies, forced LLM call
"""

import json
import os
import re
import time
import random

from agents import load_agent
from environment import load_environment
from common.registry import registry
from utils.logging.logger import TaskLogger
from utils.logging.agent_logger import AgentLogger
from .base_task import BaseTask
from .rl_module_hierarchical import (
    HierarchicalRLAgent,
    PhaseDetector,
    EpisodeTracker,
    ExperimentDataSaver,
    PHASE_EXPLORATION,
    PHASE_EXECUTION,
    PHASE_TRANSITION,
    THRESHOLD_MAP,
    TRANSITION_THRESHOLD,
)

# Rule-based threshold mapping (ablation B2: no DQN)
RULE_THRESHOLD = {
    PHASE_EXPLORATION: 0.6,
    PHASE_EXECUTION:   0.1,
    PHASE_TRANSITION:  1.5,
}

logger = AgentLogger(__name__)

# Module-level Semi-MDP agent (fresh start each run)
_global_hier_agent = HierarchicalRLAgent(state_dim=7, action_dim=3)

try:
    from autool.utils.call_model import reset_token_stats, get_token_stats
    _TOKEN_TRACKING = True
except ImportError:
    _TOKEN_TRACKING = False
    def reset_token_stats(): pass
    def get_token_stats():
        return {"input_tokens": 0, "output_tokens": 0, "n_calls": 0}


@registry.register_task("webshop_hierarchical")
class EvalWebshopHierarchical(BaseTask):

    def __init__(self,
                 llm_name="gpt",
                 llm_config=None,
                 agent_name="GPTAgent",
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
        self.context_length = llm_config.get("context_length") if llm_config else None
        self.baseline_dir   = baseline_dir

        # Label path for test tasks
        self.label_path = env_config.get("label_path", None)

        # Ablation mode
        self.ablation = agent_config.get("ablation", None)

        # ── Configurable HierInertia parameters ──
        self.phase_w_stag    = agent_config.get("phase_w_stag", 4)
        self.phase_w_trans   = agent_config.get("phase_w_trans", 2)
        self.phase_w_explore = agent_config.get("phase_w_explore", 7)
        self.phase_spike     = agent_config.get("phase_spike", 0.05)

        tmap_cfg = agent_config.get("threshold_map", None)
        if tmap_cfg and isinstance(tmap_cfg, list) and len(tmap_cfg) == 3:
            self.threshold_map = {0: tmap_cfg[0], 1: tmap_cfg[1], 2: tmap_cfg[2]}
        else:
            self.threshold_map = None
        self.transition_threshold = agent_config.get("transition_threshold", None)

        # Reward coefficients
        self.reward_progress_coeff = agent_config.get("reward_progress_coeff", 10.0)
        self.reward_token_coeff    = agent_config.get("reward_token_coeff", 3.0)
        self.reward_inertia_coeff  = agent_config.get("reward_inertia_coeff", 1.5)
        self.reward_done_bonus     = agent_config.get("reward_done_bonus", 10.0)
        self.reward_tok_baseline   = agent_config.get("reward_tok_baseline", 2000.0)

        # Re-initialize global RL agent with custom thresholds
        global _global_hier_agent
        _global_hier_agent = HierarchicalRLAgent(
            state_dim=7, action_dim=3,
            threshold_map=self.threshold_map,
            transition_threshold=self.transition_threshold,
        )

        self.agentboard = TaskLogger(
            task_name="webshop",
            log_path=log_path,
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir,
        )

    # ── Label loading ──────────────────────────────────────────────────────

    def _load_labels(self):
        """Load test task labels from JSONL file."""
        labels = []
        path = self.label_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"WebShop label file not found: {path}")
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(json.loads(line))
        return labels

    # ── Core episode evaluation ────────────────────────────────────────────

    def evaluate_env(self, index, session_id, goal, difficulty):
        """Run one WebShop episode with HierInertia."""

        # Initial reset to get first observation
        obs, reward, done, sub_reward, grounding = self.env.step(session_id, "reset[]")
        self.agent.reset(goal=goal, init_obs=obs)

        last_sub_reward = 0.0
        grounding_acc_count = 0
        score_change_record = []
        inertia_calls = 0

        logger.info("Step {:02} - Observation: {}".format(0, obs))
        trajectory = [{"Goal": goal, "id": 0}, {"Observation": obs, "id": 0}]

        # ── Phase-level tracking ──────────────────────────────────────────
        detector         = PhaseDetector(
            w_stag=self.phase_w_stag, w_trans=self.phase_w_trans,
            w_explore=self.phase_w_explore, spike=self.phase_spike,
        )
        phase_tokens     = 0
        phase_step_count = 0
        phase_inertia_ok = 0

        tracker = EpisodeTracker(index, task_name="webshop", max_steps=self.max_num_steps)
        self._current_tracker = tracker

        def _build_state(stats, cumulative_progress, step):
            return HierarchicalRLAgent.build_state(
                phase_type          = stats["phase_type"],
                phase_progress_gain = stats["phase_progress_gain"],
                phase_inertia_ratio = stats["phase_inertia_ratio"],
                phase_length        = stats["phase_length"],
                cumulative_progress = cumulative_progress,
                remaining_steps     = self.max_num_steps - step,
                max_steps           = self.max_num_steps,
            )

        # Fire macro-controller for initial phase
        init_stats = detector.get_phase_stats()
        init_state = _build_state(init_stats, 0.0, 0)
        if self.ablation == "rule_phase":
            cur_thresh = RULE_THRESHOLD[PHASE_EXPLORATION]
        else:
            cur_thresh = _global_hier_agent.on_phase_start(init_state)
        if hasattr(self.agent, 'inertia_threshold'):
            self.agent.inertia_threshold = cur_thresh
        mode_tag = f"ablation={self.ablation}" if self.ablation else "full"
        print(f"🏛️  [HIER-WS] Start | Phase=Exploration | threshold={cur_thresh:.2f} | "
              f"mode={mode_tag}", flush=True)

        for i in range(self.max_num_steps):
            # Apply threshold for current phase
            current_phase = detector.current_phase
            if current_phase == PHASE_TRANSITION:
                if hasattr(self.agent, 'inertia_threshold'):
                    self.agent.inertia_threshold = _global_hier_agent.transition_threshold
                active_thresh = _global_hier_agent.transition_threshold
            else:
                if hasattr(self.agent, 'inertia_threshold'):
                    self.agent.inertia_threshold = cur_thresh
                active_thresh = cur_thresh

            # ── HierInertia v2: Phase-conditioned memory & dynamic inertia_max ──
            if current_phase == PHASE_EXECUTION:
                mem = self.agent.memory.memory
                if len(mem) > 8:
                    self.agent.memory.memory = mem[:2] + mem[-6:]
                if hasattr(self.agent, 'inertia_max'):
                    self.agent.inertia_max = 4
            elif current_phase == PHASE_EXPLORATION:
                if hasattr(self.agent, 'inertia_max'):
                    self.agent.inertia_max = 1

            # ── Agent step ──────────────────────────────────────────────
            success, action = self.agent.run()

            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action": action, "id": i})

            if not success:
                fail_stats = detector.get_phase_stats()
                fail_state = _build_state(fail_stats, last_sub_reward, i)
                _global_hier_agent.on_phase_end(fail_state, -1.0, True)
                break

            # Execute action in environment
            obs, reward, done, sub_reward, grounding = self.env.step(session_id, action)

            if grounding:
                grounding_acc_count += 1

            logger.info("Step {:02} - Observation: {}".format(i, obs))
            logger.info("Step {:02} - Sub-reward: {}\n".format(i, sub_reward))
            trajectory.append({"Observation": obs, "id": i})
            trajectory.append({"Progress Rate": sub_reward, "id": i})

            progress_delta = sub_reward - last_sub_reward
            if sub_reward > last_sub_reward:
                score_change_record.append((i, sub_reward))

            # ── Inertia tracking ────────────────────────────────────────
            step_tokens  = (self.agent.token_counts.get("input_tokens", 0)
                            if hasattr(self.agent, "token_counts") else 0)
            inertia_used = False
            if hasattr(self.agent, 'inertia_count'):
                new_count = self.agent.inertia_count
                if new_count > inertia_calls:
                    inertia_used = True
                    if progress_delta > 0:
                        phase_inertia_ok += 1
                inertia_calls = new_count

            phase_tokens     += step_tokens
            phase_step_count += 1

            # ── Record step data ────────────────────────────────────────
            inertia_success = inertia_used and progress_delta > 0
            tracker.log_step(
                step=i, action=action, progress=sub_reward,
                progress_delta=progress_delta,
                phase=detector.current_phase, threshold=active_thresh,
                inertia_used=inertia_used, inertia_success=inertia_success,
                step_tokens=step_tokens,
            )

            # ── PhaseDetector update ────────────────────────────────────
            new_phase, phase_changed = detector.update(
                progress_delta=progress_delta,
                inertia_used=inertia_used,
                step_tokens=step_tokens,
            )

            # ── Ablation: flat_dqn fires DQN every step ─────────────────
            if self.ablation == "flat_dqn":
                flat_state  = _build_state(detector.get_phase_stats(), sub_reward, i + 1)
                flat_action = _global_hier_agent.choose_action(flat_state)
                cur_thresh  = _global_hier_agent.threshold_map.get(flat_action, 0.3)
                step_r = (10.0 if done else
                          progress_delta * 5.0 if progress_delta > 0 else -0.1)
                step_r -= step_tokens / 10000.0
                _global_hier_agent.store_transition(
                    flat_state, flat_action, step_r, flat_state, done)
                _global_hier_agent.learn()

            if phase_changed or done:
                if self.ablation != "flat_dqn":
                    p_iner_ratio = phase_inertia_ok / max(phase_step_count, 1)
                    avg_tok      = phase_tokens / max(phase_step_count, 1)
                    tok_ratio    = avg_tok / self.reward_tok_baseline
                    phase_reward = (
                        (sub_reward - last_sub_reward if phase_step_count <= 1 else
                         sum(d for d in [progress_delta] if d > 0)) * self.reward_progress_coeff
                        - max(0.0, tok_ratio - 1.0) * self.reward_token_coeff
                        + p_iner_ratio * self.reward_inertia_coeff
                        + (self.reward_done_bonus if done else 0.0)
                    )

                    next_stats = detector.get_phase_stats()
                    next_state = _build_state(next_stats, sub_reward, i + 1)
                    _global_hier_agent.on_phase_end(next_state, phase_reward, done)

                tracker.log_phase_end(
                    step=i + 1, phase_type=current_phase,
                    phase_reward=phase_reward if self.ablation != "flat_dqn" else None,
                    progress_at_end=sub_reward,
                    dqn_action=_global_hier_agent.current_action,
                    threshold=cur_thresh,
                )

                # Reset phase accumulators
                phase_tokens     = 0
                phase_step_count = 0
                phase_inertia_ok = 0

                if not done and phase_changed:
                    if self.ablation == "rule_phase":
                        cur_thresh = RULE_THRESHOLD.get(new_phase, 0.3)
                    elif self.ablation != "flat_dqn":
                        next_stats = detector.get_phase_stats()
                        next_state = _build_state(next_stats, sub_reward, i + 1)
                        cur_thresh = _global_hier_agent.on_phase_start(next_state)
                    print(f"🏛️  [HIER-WS] Step {i} | PhaseChange → {new_phase} | "
                          f"threshold={cur_thresh:.2f}", flush=True)

            if done:
                # WebShop: success = reward == 1.0 (full score)
                is_success = (reward >= 1.0)
                env_details = {"task_name": "webshop", "goal": goal, "difficulty": difficulty}
                self.agentboard.log_example(
                    index, is_success, sub_reward,
                    grounding_acc_count / (i + 1),
                    score_change_record, env_details, trajectory,
                )
                return sub_reward, is_success, grounding_acc_count / (i + 1), score_change_record, i

            last_sub_reward = sub_reward
            self.agent.update(action=action, state=obs)

        # ── Max steps exceeded ──────────────────────────────────────────
        env_details = {"task_name": "webshop", "goal": goal, "difficulty": difficulty}
        try:
            example_prompt = self.agent.get_example_prompt()
        except Exception:
            example_prompt = None

        self.agentboard.log_example(
            index, False, last_sub_reward,
            grounding_acc_count / max(i + 1, 1),
            score_change_record, env_details, trajectory,
            example_prompt,
        )
        return last_sub_reward, False, grounding_acc_count / max(i + 1, 1), score_change_record, i

    # ── Checkpoint / Resume helpers ────────────────────────────────────────

    def _checkpoint_path(self):
        return os.path.join(self._log_path_dir or "", "checkpoint.jsonl")

    def _load_completed_indices(self):
        completed = {}
        ckpt = self._checkpoint_path()
        if os.path.exists(ckpt):
            with open(ckpt, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        completed[rec["index"]] = rec
                    except Exception:
                        continue
        return completed

    def _save_checkpoint(self, index, done, score, grounding_acc, score_change_record,
                         num_steps, tok, difficulty):
        ckpt = self._checkpoint_path()
        rec = {
            "index": index,
            "done": done,
            "score": score,
            "grounding_acc": grounding_acc,
            "score_change_record": score_change_record,
            "num_steps": num_steps,
            "tokens": tok,
            "difficulty": difficulty,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(ckpt, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ── Main evaluation loop with resume ──────────────────────────────────

    def evaluate(self):
        # Load labels and limit to num_exam if set in run_config (handled externally via env)
        labels = self._load_labels()

        # Load the offline WebShop environment (BM25 index built once here)
        self.env = load_environment("webshop", self.env_cfg)

        # Limit to num_exam tasks
        labels = labels[:self.num_exam]
        print(f"[WebShop] Running {len(labels)} tasks (num_exam={self.num_exam})", flush=True)

        # Resume support
        completed = self._load_completed_indices()
        if completed:
            print(f"[RESUME] Found {len(completed)} completed examples, will skip them.")

        # Data saver for paper figures
        exp_tag = os.path.basename(self._log_path_dir) if self._log_path_dir else "ws_default"
        data_saver = ExperimentDataSaver(exp_tag)

        scores, srs, grounding_accs, score_state_records, difficulties = [], [], [], [], []

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
                print(f"[RESUME] Skipping example {index} (SR={c['done']}, PR={c['score']:.3f})")
                continue

            logger.goal(f"Example {index} | Goal: {goal} | Difficulty: {difficulty}")
            reset_token_stats()
            try:
                score, done, grounding_acc, score_change_record, num_steps = \
                    self.evaluate_env(index, session_id, goal, difficulty)
            except Exception as e:
                print(f"[ERROR] Example {index} failed: {e}", flush=True)
                score, done, grounding_acc, score_change_record, num_steps = 0.0, False, 0.0, [], 0
            tok = get_token_stats()

            difficulties.append(difficulty)
            logger.finish(f"Example {index} | Success: {done} , PR: {score:.3f} , Steps: {num_steps + 1}\n")
            print(f"[TOKEN] Example {index}: in={tok['input_tokens']}, "
                  f"out={tok['output_tokens']}, calls={tok['n_calls']}", flush=True)

            self._save_checkpoint(index, done, score, grounding_acc,
                                  score_change_record, num_steps + 1, tok, difficulty)

            if hasattr(self, '_current_tracker') and self._current_tracker is not None:
                data_saver.add_episode(
                    self._current_tracker, done, score,
                    tok.get('input_tokens', 0) + tok.get('output_tokens', 0),
                    tok.get('n_calls', 0))
                self._current_tracker = None

            srs.append(1.0 if done else 0.0)
            scores.append(score)
            grounding_accs.append(grounding_acc)
            score_state_records.append(score_change_record)

        sr = sum(srs)    / len(srs)
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

        try:
            data_saver.save_all()
            _global_hier_agent.save_rl_curve(tag=exp_tag)
        except Exception as e:
            print(f"[WARN] Failed to save experiment data: {e}")

        return srs, scores, grounding_accs, score_state_records, easy_sr, hard_sr, easy_pr, hard_pr

    @classmethod
    def from_config(cls, run_config, llm_config, agent_config, env_config, llm=None):
        return cls(
            llm_name     = llm_config.get("name", "gpt"),
            llm_config   = llm_config,
            agent_name   = agent_config.get("name", "ReactInertiaAgent"),
            agent_config = agent_config,
            env_config   = env_config,
            run_config   = run_config,
            llm          = llm,
            baseline_dir = run_config.get("baseline_dir", "data/baseline_results"),
            log_path     = run_config.get("log_path", None),
        )
