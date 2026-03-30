"""
scienceworld_hierarchical.py — ScienceWorld task with HierInertia (Semi-MDP).

Registers as task type "scienceworld_hierarchical".

Architecture mirrors alfworld_hierarchical.py but adapts to ScienceWorld's
evaluate_env() signature and step API.
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
    PHASE_EXPLORATION: 0.6,   # cautious
    PHASE_EXECUTION:   0.1,   # aggressive
    PHASE_TRANSITION:  1.5,   # forced LLM
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


@registry.register_task("scienceworld_hierarchical")
class EvalScienceworldHierarchical(BaseTask):

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
        self.simplefied = env_config.get("simplefied", False)
        seed = env_config.get("seed", 42)
        self.set_seed(seed)
        self.simplification_str = self.build_simplification_str()
        self.env_cfg = env_config

        self.max_num_steps  = int(run_config.get("max_num_steps", 30))
        self._log_path_dir  = log_path  # for resume support
        self.context_length = llm_config.get("context_length")
        self.baseline_dir   = baseline_dir

        # Ablation mode: "rule_phase" or "flat_dqn" or None (full HierInertia)
        self.ablation = agent_config.get("ablation", None)

        # ── Configurable HierInertia parameters (from YAML agent config) ──
        self.phase_w_stag    = agent_config.get("phase_w_stag", 4)
        self.phase_w_trans   = agent_config.get("phase_w_trans", 2)
        self.phase_w_explore = agent_config.get("phase_w_explore", 7)
        self.phase_spike     = agent_config.get("phase_spike", 0.05)

        # Threshold map: YAML can override as list [cautious, balanced, aggressive]
        tmap_cfg = agent_config.get("threshold_map", None)
        if tmap_cfg and isinstance(tmap_cfg, list) and len(tmap_cfg) == 3:
            self.threshold_map = {0: tmap_cfg[0], 1: tmap_cfg[1], 2: tmap_cfg[2]}
        else:
            self.threshold_map = None  # use default
        self.transition_threshold = agent_config.get("transition_threshold", None)

        # Reward coefficients
        self.reward_progress_coeff = agent_config.get("reward_progress_coeff", 10.0)
        self.reward_token_coeff    = agent_config.get("reward_token_coeff", 2.0)
        self.reward_inertia_coeff  = agent_config.get("reward_inertia_coeff", 1.5)
        self.reward_done_bonus     = agent_config.get("reward_done_bonus", 10.0)
        self.reward_tok_baseline   = agent_config.get("reward_tok_baseline", 2000.0)

        # AB-3 ablation: disable phase-conditioned memory trimming
        self.phase_memory_trim = agent_config.get("phase_memory_trim", True)

        # Level-2 StaticPhase ablation: disable dynamic inertia_max switching
        self.phase_dynamic_inertia = agent_config.get("phase_dynamic_inertia", True)

        # Re-initialize global RL agent with custom thresholds if provided
        global _global_hier_agent
        _global_hier_agent = HierarchicalRLAgent(
            state_dim=7, action_dim=3,
            threshold_map=self.threshold_map,
            transition_threshold=self.transition_threshold,
        )

        self.agentboard = TaskLogger(
            task_name="scienceworld",
            log_path=log_path,
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir,
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

        reward      = 0.0
        last_reward = 0.0
        isDone      = False
        grounding_acc_count = 0
        score_change_record = []
        inertia_calls = 0

        logger.info("Step {:02} - Observation: {}".format(0, init_obs))

        trajectory = []
        trajectory.append({"Goal": modified_goal, "id": 0})
        trajectory.append({"Observation": init_obs, "id": 0})

        # ── Phase-level tracking ──────────────────────────────────
        detector         = PhaseDetector(w_stag=self.phase_w_stag, w_trans=self.phase_w_trans,
                                         w_explore=self.phase_w_explore, spike=self.phase_spike)
        phase_tokens     = 0
        phase_step_count = 0
        phase_inertia_ok = 0

        # ── Structured data collection ──
        tracker = EpisodeTracker(index, task_name=task_name, max_steps=self.max_num_steps)
        self._current_tracker = tracker  # expose to evaluate() for collection

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
        # [P0-2] 初始 phase=EXPLORATION → 全量 prompt（prompt_window=None）
        if hasattr(self.agent, 'prompt_window'):
            self.agent.prompt_window = None
        mode_tag = f"ablation={self.ablation}" if self.ablation else "full"
        print(f"🏛️  [HIER] Start | Phase=Exploration | threshold={cur_thresh:.2f} | "
              f"mode={mode_tag} | epsilon={_global_hier_agent.epsilon:.3f}", flush=True)

        for i in range(self.max_num_steps):
            # Apply threshold (override to TRANSITION_THRESHOLD during Transition)
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
                # Execution: trim memory to save tokens (keep head + recent context)
                if self.phase_memory_trim:
                    mem = self.agent.memory.memory
                    if len(mem) > 8:
                        self.agent.memory.memory = mem[:2] + mem[-6:]
                # Allow more consecutive inertia in execution phase
                if self.phase_dynamic_inertia and hasattr(self.agent, 'inertia_max'):
                    self.agent.inertia_max = 4
            elif current_phase == PHASE_EXPLORATION:
                if self.phase_dynamic_inertia and hasattr(self.agent, 'inertia_max'):
                    self.agent.inertia_max = 1

            # ── Agent step ────────────────────────────────────────
            success, action = self.agent.run()

            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action": action, "id": i})

            if not success:
                fail_stats = detector.get_phase_stats()
                fail_state = _build_state(fail_stats, last_reward, i)
                _global_hier_agent.on_phase_end(fail_state, -1.0, True)
                break

            observation, reward, isDone, info = self.env.step(action)
            if action in self.env.get_action_space(abstract=False):
                grounding_acc_count += 1

            logger.info("Step {:02} - Observation: {}".format(i, observation))
            logger.info("Step {:02} - Progress Rate: {}\n".format(i, reward))
            trajectory.append({"Observation": observation, "id": i})
            trajectory.append({"Progress Rate": reward,   "id": i})

            progress_delta = reward - last_reward
            if reward > last_reward:
                score_change_record.append((i, reward))

            # ── Inertia tracking ──────────────────────────────────
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

            # ── Record step data ──────────────────────────────────
            inertia_success = inertia_used and progress_delta > 0
            tracker.log_step(
                step=i, action=action, progress=reward,
                progress_delta=progress_delta,
                phase=detector.current_phase, threshold=active_thresh,
                inertia_used=inertia_used, inertia_success=inertia_success,
                step_tokens=step_tokens,
            )

            # ── PhaseDetector update ──────────────────────────────
            new_phase, phase_changed = detector.update(
                progress_delta=progress_delta,
                inertia_used=inertia_used,
                step_tokens=step_tokens,
            )

            # ── Ablation: flat_dqn fires DQN every step ─────────
            if self.ablation == "flat_dqn":
                flat_state = _build_state(detector.get_phase_stats(), reward, i + 1)
                flat_action = _global_hier_agent.choose_action(flat_state)
                cur_thresh = _global_hier_agent.threshold_map.get(flat_action, 0.3)
                # step-level reward
                step_r = (10.0 if isDone else
                          progress_delta * 5.0 if progress_delta > 0 else -0.1)
                step_r -= step_tokens / 10000.0
                _global_hier_agent.store_transition(
                    flat_state, flat_action, step_r, flat_state, isDone)
                _global_hier_agent.learn()

            if phase_changed or isDone:
                # Phase-level reward (skip for flat_dqn ablation)
                if self.ablation != "flat_dqn":
                    p_iner_ratio = phase_inertia_ok / max(phase_step_count, 1)
                    avg_tok      = phase_tokens / max(phase_step_count, 1)
                    tok_ratio    = avg_tok / self.reward_tok_baseline
                    phase_reward = (
                        (reward - last_reward if phase_step_count <= 1 else
                         sum(d for d in [progress_delta] if d > 0)) * self.reward_progress_coeff
                        - max(0.0, tok_ratio - 1.0) * self.reward_token_coeff
                        + p_iner_ratio * self.reward_inertia_coeff
                        + (self.reward_done_bonus if isDone else 0.0)
                    )

                    next_stats = detector.get_phase_stats()
                    next_state = _build_state(next_stats, reward, i + 1)
                    _global_hier_agent.on_phase_end(next_state, phase_reward, isDone)

                # Record completed phase
                tracker.log_phase_end(
                    step=i + 1, phase_type=current_phase,
                    phase_reward=phase_reward if self.ablation != "flat_dqn" else None,
                    progress_at_end=reward,
                    dqn_action=_global_hier_agent.current_action,
                    threshold=cur_thresh,
                )

                # Reset phase accumulators
                phase_tokens     = 0
                phase_step_count = 0
                phase_inertia_ok = 0

                if not isDone and phase_changed:
                    if self.ablation == "rule_phase":
                        cur_thresh = RULE_THRESHOLD.get(new_phase, 0.3)
                    elif self.ablation != "flat_dqn":
                        next_stats = detector.get_phase_stats()
                        next_state = _build_state(next_stats, reward, i + 1)
                        cur_thresh = _global_hier_agent.on_phase_start(next_state)
                    # [P0-2] phase 切换时更新 prompt_window：
                    # EXECUTION → 短窗口(8条)，压缩 token，加速单步；
                    # TRANSITION → 中窗口(14条)，保留近期关键上下文；
                    # EXPLORATION → 全量(None)，LLM 需要完整历史做规划。
                    if hasattr(self.agent, 'prompt_window'):
                        _pw_map = {
                            PHASE_EXECUTION:   8,
                            PHASE_TRANSITION:  14,
                            PHASE_EXPLORATION: None,
                        }
                        self.agent.prompt_window = _pw_map.get(new_phase, None)
                    print(f"🏛️  [HIER] Step {i} | PhaseChange → {new_phase} | "
                          f"threshold={cur_thresh:.2f} | "
                          f"prompt_window={getattr(self.agent, 'prompt_window', 'N/A')} | "
                          f"epsilon={_global_hier_agent.epsilon:.3f}", flush=True)

            if isDone:
                env_details = {"task_name": task_name, "goal": self.agent.goal,
                               "difficulty": self.env.difficulty}
                self.agentboard.log_example(
                    index, True, 1.0,
                    grounding_acc_count / (i + 1),
                    score_change_record, env_details, trajectory,
                )
                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

            last_reward = reward
            self.agent.update(action=action, state=observation)

        # ── Max steps exceeded ────────────────────────────────────
        env_details = {"task_name": task_name, "goal": self.agent.goal,
                       "difficulty": self.env.difficulty}
        try:
            example_prompt = self.agent.get_example_prompt()
        except Exception:
            example_prompt = None

        progress_rate = reward
        self.agentboard.log_example(
            index, isDone, progress_rate,
            grounding_acc_count / (i + 1),
            score_change_record, env_details, trajectory,
            example_prompt,
        )
        return progress_rate, isDone, grounding_acc_count / (i + 1), score_change_record, i

    # ── Checkpoint / Resume helpers ──────────────────────────────────

    def _checkpoint_path(self):
        return os.path.join(self._log_path_dir or "", "checkpoint.jsonl")

    def _load_completed_indices(self):
        """Load already-completed example indices from checkpoint for resume."""
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
        """Append one example result to checkpoint (append-only, crash-safe)."""
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

    # ── Main evaluation loop with resume ──────────────────────────

    def evaluate(self):
        self.env = load_environment("scienceworld", self.env_cfg)
        labels   = self.env.labels
        scores, score_state_records, grounding_accs, srs, difficulties = [], [], [], [], []

        # Resume support: load checkpoint
        completed = self._load_completed_indices()
        if completed:
            print(f"[RESUME] Found {len(completed)} completed examples in checkpoint, will skip them.")

        # Data saver for paper figures
        exp_tag = os.path.basename(self._log_path_dir) if self._log_path_dir else "sw_default"
        data_saver = ExperimentDataSaver(exp_tag)

        for index, (k, v) in enumerate(labels.items()):
            task_name     = v["task_name"]
            var           = v["var"]
            modified_goal = v["modified_goal"]

            # Skip already-completed examples (from checkpoint)
            if index in completed:
                c = completed[index]
                srs.append(1.0 if c["done"] else 0.0)
                scores.append(c["score"])
                grounding_accs.append(c["grounding_acc"])
                score_state_records.append(c.get("score_change_record", []))
                difficulties.append(c.get("difficulty", "easy"))
                print(f"[RESUME] Skipping example {index} (SR={c['done']}, PR={c['score']:.3f})")
                continue

            logger.goal("Example {} | Goal: {}".format(
                index, f"task_name: {task_name}, var: {var}, {modified_goal}"))
            reset_token_stats()
            try:
                score, done, grounding_acc, score_change_record, num_steps = \
                    self.evaluate_env(index, task_name, var, modified_goal)
            except Exception as e:
                print(f"[ERROR] Example {index} failed: {e}", flush=True)
                # Save as failed but don't crash — allow remaining examples to run
                score, done, grounding_acc, score_change_record, num_steps = 0.0, False, 0.0, [], 0
            tok = get_token_stats()

            difficulty = self.env.difficulty if hasattr(self.env, 'difficulty') else "easy"
            difficulties.append(difficulty)
            logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(
                index, done, score, num_steps + 1))
            print(f"[TOKEN] Example {index}: in={tok['input_tokens']}, "
                  f"out={tok['output_tokens']}, calls={tok['n_calls']}", flush=True)

            # Save checkpoint immediately (crash-safe)
            self._save_checkpoint(index, done, score, grounding_acc,
                                  score_change_record, num_steps + 1, tok, difficulty)

            # Collect episode data for paper figures
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

        hard_sr = sum(hard_sr) / len(hard_sr) if hard_sr else 0
        hard_pr = sum(hard_pr) / len(hard_pr) if hard_pr else 0
        easy_sr = sum(easy_sr) / len(easy_sr) if easy_sr else 0
        easy_pr = sum(easy_pr) / len(easy_pr) if easy_pr else 0

        self.agentboard.log_summary(sr, pr, gr, score_state_records,
                                    hard_sr, hard_pr, easy_sr, easy_pr)

        # Save experiment data for paper figures
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
