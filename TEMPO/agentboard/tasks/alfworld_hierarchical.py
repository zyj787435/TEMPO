"""
alfworld_hierarchical.py — AlfWorld task with HierInertia (Semi-MDP).

Registers as task type "alfworld_hierarchical".

Architecture:
  - PhaseDetector detects Exploration / Execution / Transition phases
  - HierarchicalRLAgent fires at phase boundaries (Semi-MDP)
  - Within a phase, inertia threshold is held fixed (set by macro-controller)
  - Phase-level reward = progress_gain*10 − token_excess*2 + inertia_efficiency*1.5
"""

import numpy as np
import time
import json
import copy
import os
from datetime import datetime

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
    PHASE_TRANSITION,
    TRANSITION_THRESHOLD,
    PHASE_EXPLORATION,
    PHASE_EXECUTION,
    THRESHOLD_MAP,
)

logger = AgentLogger(__name__)

try:
    from autool.utils.call_model import reset_token_stats, get_token_stats
except ImportError:
    def reset_token_stats(): pass
    def get_token_stats():
        return {"input_tokens": 0, "output_tokens": 0, "n_calls": 0}

# Module-level Semi-MDP agent (fresh start each run)
_global_hier_agent = HierarchicalRLAgent(state_dim=7, action_dim=3)

prefixes = {
    'pick_and_place'       : 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place' : 'heat',
    'pick_cool_then_place' : 'cool',
    'look_at_obj'          : 'examine',
    'pick_two_obj'         : 'puttwo',
}


@registry.register_task("alfworld_hierarchical")
class EvalalfworldHierarchical(BaseTask):

    def __init__(self,
                 llm_config=None,
                 agent_name='agent_name',
                 max_num_steps=50,
                 num_exams=134,
                 init_prompt_path='prompts/alfworld_base.json',
                 agent_config=None,
                 env_config=None,
                 llm=None,
                 baseline_dir=None,
                 log_path=None):
        super().__init__()
        self.agent = load_agent(agent_name, agent_config, llm)
        with open(init_prompt_path, 'r') as f:
            self.prompts = json.load(f)
        self.env_cfg       = env_config
        self.max_num_steps = max_num_steps
        self.num_exams     = num_exams
        self.baseline_dir  = baseline_dir
        self.log_path      = log_path

        # ── Configurable HierInertia parameters (from YAML agent config) ──
        self.phase_w_stag    = agent_config.get("phase_w_stag", 5)
        self.phase_w_trans   = agent_config.get("phase_w_trans", 2)
        self.phase_w_explore = agent_config.get("phase_w_explore", 8)
        self.phase_spike     = agent_config.get("phase_spike", 0.05)

        tmap_cfg = agent_config.get("threshold_map", None)
        if tmap_cfg and isinstance(tmap_cfg, list) and len(tmap_cfg) == 3:
            self.threshold_map = {0: tmap_cfg[0], 1: tmap_cfg[1], 2: tmap_cfg[2]}
        else:
            self.threshold_map = None
        self.transition_threshold = agent_config.get("transition_threshold", None)

        self.reward_progress_coeff = agent_config.get("reward_progress_coeff", 10.0)
        self.reward_token_coeff    = agent_config.get("reward_token_coeff", 2.0)
        self.reward_inertia_coeff  = agent_config.get("reward_inertia_coeff", 1.5)
        self.reward_done_bonus     = agent_config.get("reward_done_bonus", 10.0)
        self.reward_tok_baseline   = agent_config.get("reward_tok_baseline", 2000.0)

        global _global_hier_agent
        _global_hier_agent = HierarchicalRLAgent(
            state_dim=7, action_dim=3,
            threshold_map=self.threshold_map,
            transition_threshold=self.transition_threshold,
        )

        self.agentboard    = TaskLogger(
            task_name="alfworld",
            log_path=log_path,
            max_num_steps=self.max_num_steps,
            baseline_dir=self.baseline_dir,
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
        goal    = ob.split('\n')[1].split("Your task is to:")[1].strip()
        self.agent.reset(goal=goal, init_obs=init_ob)
        logger.goal("Example {} | Goal: {}".format(index, self.agent.goal))
        init_prompt_dict = copy.deepcopy(self.prompts)
        init_prompt_dict['examples'] = examples

        reward      = 0.0
        last_reward = 0.0
        done        = False
        grounding_acc_count = 0
        score_change_record = []

        # Phase-tracking
        detector    = PhaseDetector(w_stag=self.phase_w_stag, w_trans=self.phase_w_trans,
                                     w_explore=self.phase_w_explore, spike=self.phase_spike)
        inertia_calls = 0

        # Phase-level accumulators for reward
        phase_tokens      = 0
        phase_step_count  = 0
        phase_inertia_ok  = 0   # inertia steps that produced progress

        # ── Structured data collection ──
        tracker = EpisodeTracker(index, task_name=goal, max_steps=self.max_num_steps)
        self._current_tracker = tracker

        # Build initial state (phase not yet started)
        def _build_state(stats, cumulative_progress, step):
            return HierarchicalRLAgent.build_state(
                phase_type        = stats["phase_type"],
                phase_progress_gain = stats["phase_progress_gain"],
                phase_inertia_ratio = stats["phase_inertia_ratio"],
                phase_length      = stats["phase_length"],
                cumulative_progress = cumulative_progress,
                remaining_steps   = self.max_num_steps - step,
                max_steps         = self.max_num_steps,
            )

        # Fire macro-controller for initial phase
        init_stats  = detector.get_phase_stats()
        init_state  = _build_state(init_stats, 0.0, 0)
        cur_thresh  = _global_hier_agent.on_phase_start(init_state)
        if hasattr(self.agent, 'inertia_threshold'):
            self.agent.inertia_threshold = cur_thresh
        print(f"🏛️  [HIER] Start | Phase=Exploration | threshold={cur_thresh:.2f} | "
              f"epsilon={_global_hier_agent.epsilon:.3f}", flush=True)

        logger.info("Step {:02} - Message: {}".format(0, init_ob))
        trajectory = [{"Goal": goal, "Observation": init_ob, "id": 0}]

        for i in range(self.max_num_steps):
            # Apply threshold (overridden to TRANSITION_THRESHOLD during Transition)
            current_phase = detector.current_phase
            if current_phase == PHASE_TRANSITION:
                if hasattr(self.agent, 'inertia_threshold'):
                    self.agent.inertia_threshold = _global_hier_agent.transition_threshold
                active_thresh = _global_hier_agent.transition_threshold
            else:
                if hasattr(self.agent, 'inertia_threshold'):
                    self.agent.inertia_threshold = cur_thresh
                active_thresh = cur_thresh

            # ── Agent step ────────────────────────────────────────
            success, action = self.agent.run(init_prompt_dict=init_prompt_dict)
            if not success:
                # terminal failure — close out phase
                fail_stats  = detector.get_phase_stats()
                fail_state  = _build_state(fail_stats, last_reward, i)
                _global_hier_agent.on_phase_end(fail_state, -1.0, True)
                break

            action = self.parseAction(action)
            if action in self.env.get_action_space():
                grounding_acc_count += 1.0

            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action": action, "id": i})

            observation, reward, done, info = self.env.step(action)
            logger.info("Step {:02} - Observation: {}".format(i, observation))
            logger.info("Step {:02} - Progress Rate: {}\n".format(i, reward))
            trajectory.append({"Observation": observation,
                                "Progress Rate": reward, "id": i})

            progress_delta = reward - last_reward
            if reward > last_reward:
                score_change_record.append((i, reward))

            # ── Inertia tracking ──────────────────────────────────
            step_tokens = (self.agent.token_counts.get("input_tokens", 0)
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

            if phase_changed or done:
                # Compute phase-level reward
                prev_stats      = detector.get_phase_stats()
                # Use completed-phase stats (before reset inside detector._switch)
                # Note: detector._reset_phase_stats was already called, so use accumulators
                p_gain      = max(reward - (last_reward if phase_step_count <= 1 else 0), 0.0)
                p_iner_ratio = (phase_inertia_ok / max(phase_step_count, 1))
                avg_tok      = phase_tokens / max(phase_step_count, 1)
                tok_ratio    = avg_tok / self.reward_tok_baseline
                phase_reward = (
                    p_gain * self.reward_progress_coeff
                    - max(0.0, tok_ratio - 1.0) * self.reward_token_coeff
                    + p_iner_ratio * self.reward_inertia_coeff
                    + (self.reward_done_bonus if done else 0.0)
                )

                next_stats = detector.get_phase_stats()
                next_state = _build_state(next_stats, reward, i + 1)
                _global_hier_agent.on_phase_end(next_state, phase_reward, done)

                # Record completed phase
                tracker.log_phase_end(
                    step=i + 1, phase_type=current_phase,
                    phase_reward=phase_reward,
                    progress_at_end=reward,
                    dqn_action=_global_hier_agent.current_action,
                    threshold=cur_thresh,
                )

                # Reset phase accumulators
                phase_tokens     = 0
                phase_step_count = 0
                phase_inertia_ok = 0

                if not done and phase_changed:
                    # Start new phase — macro-controller fires again
                    cur_thresh = _global_hier_agent.on_phase_start(next_state)
                    print(f"🏛️  [HIER] Step {i} | PhaseChange → {new_phase} | "
                          f"threshold={cur_thresh:.2f} | epsilon={_global_hier_agent.epsilon:.3f}",
                          flush=True)

            last_reward = reward
            self.agent.update(action=action, state=observation)

            if done:
                game_name   = self.env.cur_task_name.split('/')[0]
                env_details = {"task_name": game_name, "goal": self.agent.goal,
                               "difficulty": self.env.difficulty}
                self.agentboard.log_example(
                    index, True, reward,
                    grounding_acc_count / (i + 1),
                    score_change_record, env_details, trajectory,
                )
                elapsed     = time.time() - t0
                total_steps = i + 1
                print(f"[TIMER]  Example {index}: elapsed={elapsed:.2f}s", flush=True)
                print(f"[HIER]   Example {index}: steps={total_steps}, "
                      f"inertia={inertia_calls}", flush=True)
                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

        # ── Max steps exceeded ────────────────────────────────────
        game_name   = self.env.cur_task_name.split('/')[0]
        env_details = {"task_name": game_name, "goal": self.agent.goal,
                       "difficulty": self.env.difficulty}
        progress_rate = reward
        self.agentboard.log_example(
            index, done, progress_rate,
            grounding_acc_count / (i + 1),
            score_change_record, env_details, trajectory,
        )
        elapsed     = time.time() - t0
        total_steps = i + 1
        print(f"[TIMER]  Example {index}: elapsed={elapsed:.2f}s", flush=True)
        print(f"[HIER]   Example {index}: steps={total_steps}, "
              f"inertia={inertia_calls}", flush=True)
        return progress_rate, done, grounding_acc_count / (i + 1), score_change_record, i

    # ── Checkpoint / Resume helpers ──────────────────────────────────

    def _checkpoint_path(self):
        return os.path.join(self.log_path or "", "checkpoint.jsonl")

    def _load_completed_indices(self):
        """Load completed examples from checkpoint for resume."""
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

    def _save_checkpoint(self, index, done, score, grounding_acc,
                         score_change_record, steps, tok, difficulty):
        """Append one result to checkpoint (append-only, crash-safe)."""
        ckpt = self._checkpoint_path()
        rec = {
            "index": index,
            "done": done,
            "score": score,
            "grounding_acc": grounding_acc,
            "score_change_record": score_change_record,
            "steps": steps,
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
        self.env = load_environment('alfworld', self.env_cfg)
        scores, srs, grounding_accs, score_state_records, difficulties = [], [], [], [], []

        # Resume support
        completed = self._load_completed_indices()
        if completed:
            print(f"[RESUME] Found {len(completed)} completed examples in checkpoint, will skip them.")

        # Data saver for paper figures
        exp_tag = os.path.basename(self.log_path) if self.log_path else "aw_default"
        data_saver = ExperimentDataSaver(exp_tag)

        for id in range(self.num_exams):
            ob, info = self.env.reset()
            ob   = '\n'.join(ob[0].split('\n\n')[1:])
            name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
            difficulty = self.env.difficulty
            difficulties.append(difficulty)

            # Skip completed examples
            if id in completed:
                c = completed[id]
                srs.append(1.0 if c["done"] else 0.0)
                scores.append(c["score"])
                grounding_accs.append(c["grounding_acc"])
                score_state_records.append(c.get("score_change_record", []))
                print(f"[RESUME] Skipping example {id} (SR={c['done']}, PR={c['score']:.3f})")
                continue

            for k, v in prefixes.items():
                if name.startswith(k):
                    reset_token_stats()
                    try:
                        score, is_done, grounding_acc, score_change_record, steps = \
                            self.evaluate_env(ob=ob,
                                              examples="".join(self.prompts['examples'][v]),
                                              index=id)
                    except Exception as e:
                        print(f"[ERROR] Example {id} failed: {e}", flush=True)
                        score, is_done, grounding_acc, score_change_record, steps = 0.0, False, 0.0, [], 0
                    tok = get_token_stats()

                    srs.append(1.0 if is_done else 0.0)
                    scores.append(score)
                    grounding_accs.append(grounding_acc)
                    score_state_records.append(score_change_record)
                    logger.finish(
                        "Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(
                            id, is_done, score, steps))
                    print(f"[TOKEN] Example {id}: in={tok['input_tokens']}, "
                          f"out={tok['output_tokens']}, calls={tok['n_calls']}", flush=True)

                    # Save checkpoint immediately
                    self._save_checkpoint(id, is_done, score, grounding_acc,
                                          score_change_record, steps, tok, difficulty)

                    # Collect episode data for paper figures
                    if hasattr(self, '_current_tracker') and self._current_tracker is not None:
                        data_saver.add_episode(
                            self._current_tracker, is_done, score,
                            tok.get('input_tokens', 0) + tok.get('output_tokens', 0),
                            tok.get('n_calls', 0))
                        self._current_tracker = None

        sr = sum(srs)    / len(srs)
        pr = sum(scores) / len(scores)
        gr = sum(grounding_accs) / len(grounding_accs)

        easy_srs = [s for s, d in zip(srs, difficulties)    if d == "easy"]
        hard_srs = [s for s, d in zip(srs, difficulties)    if d == "hard"]
        easy_prs = [s for s, d in zip(scores, difficulties) if d == "easy"]
        hard_prs = [s for s, d in zip(scores, difficulties) if d == "hard"]
        easy_sr  = sum(easy_srs) / len(easy_srs) if easy_srs else 0
        hard_sr  = sum(hard_srs) / len(hard_srs) if hard_srs else 0
        easy_pr  = sum(easy_prs) / len(easy_prs) if easy_prs else 0
        hard_pr  = sum(hard_prs) / len(hard_prs) if hard_prs else 0

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
            llm_config       = llm_config,
            agent_name       = agent_config.get("name", "ReactInertiaAgent"),
            max_num_steps    = run_config.get("max_num_steps", 50),
            num_exams        = run_config.get("num_exam", 134),
            init_prompt_path = agent_config.get("init_prompt_path",
                                                "prompts/alfworld_base.json"),
            agent_config     = agent_config,
            env_config       = env_config,
            llm              = llm,
            baseline_dir     = run_config.get("baseline_dir", "data/baseline_results"),
            log_path         = run_config.get("log_path", None),
        )
