"""
rl_module_hierarchical.py — HierInertia: Hierarchical Semi-MDP for inertia scheduling.

Key differences vs AdaInertia (flat DQN):
  - DQN fires at PHASE BOUNDARIES only (Semi-MDP), not every step
  - Explicit 3-phase state machine: Exploration / Execution / Transition
  - Phase-level state (7-dim) summarizes completed phase
  - Phase-level reward: progress gain × efficiency − token cost
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
import time
from collections import deque

# ────────────────────────────────────────────────────────────────────
# Phase constants
# ────────────────────────────────────────────────────────────────────
PHASE_EXPLORATION = 0   # Cautious: low inertia rate, gathering info
PHASE_EXECUTION   = 1   # Confident: high inertia rate, exploiting TIG
PHASE_TRANSITION  = 2   # Progress spike detected: force LLM to consolidate

# Action → inertia threshold mapping
THRESHOLD_MAP = {
    0: 0.6,   # A0 = Cautious
    1: 0.3,   # A1 = Balanced
    2: 0.1,   # A2 = Aggressive
}
TRANSITION_THRESHOLD = 1.5   # always force LLM during Transition phase


# ────────────────────────────────────────────────────────────────────
# PhaseDetector  (explicit 3-state state machine)
# ────────────────────────────────────────────────────────────────────
class PhaseDetector:
    """Detects phase boundaries and maintains per-phase statistics.

    Parameters
    ----------
    w_stag   : int   — consecutive steps with no-progress → Exploration
    w_trans  : int   — steps spent in Transition before returning
    w_explore: int   — max steps in Exploration before forcing Execution
    spike    : float — progress_delta threshold for a "progress spike"
    """

    def __init__(self, w_stag=5, w_trans=2, w_explore=8, spike=0.05):
        self.w_stag    = w_stag
        self.w_trans   = w_trans
        self.w_explore = w_explore
        self.spike     = spike
        self.reset()

    def reset(self):
        self.current_phase = PHASE_EXPLORATION
        self.phase_step    = 0      # steps in current phase
        self.no_progress   = 0      # consecutive steps without progress
        self.trans_count   = 0      # countdown for TRANSITION phase exit

        # Per-phase accumulators
        self.phase_inertia_calls  = 0
        self.phase_progress_gain  = 0.0
        self.phase_token_total    = 0
        self.last_phase_progress  = 0.0   # progress at phase start

    def _reset_phase_stats(self):
        self.phase_step          = 0
        self.phase_inertia_calls = 0
        self.phase_progress_gain = 0.0
        self.phase_token_total   = 0

    def _switch(self, new_phase):
        self.current_phase = new_phase
        self._reset_phase_stats()
        self.no_progress = 0

    def update(self, progress_delta, inertia_used, step_tokens):
        """Call once per environment step.

        Returns
        -------
        current_phase : int
        phase_changed : bool — True if a phase boundary was crossed this step
        """
        prev_phase = self.current_phase
        self.phase_step          += 1
        self.phase_progress_gain += max(progress_delta, 0.0)
        self.phase_token_total   += step_tokens
        if inertia_used:
            self.phase_inertia_calls += 1

        if progress_delta <= 0:
            self.no_progress += 1
        else:
            self.no_progress = 0

        # ── TRANSITION: countdown ─────────────────────────────────
        if self.current_phase == PHASE_TRANSITION:
            self.trans_count -= 1
            if self.trans_count <= 0:
                new = PHASE_EXECUTION if progress_delta > 0 else PHASE_EXPLORATION
                self._switch(new)

        # ── EXECUTION: watch for stagnation or spike ──────────────
        elif self.current_phase == PHASE_EXECUTION:
            if progress_delta >= self.spike:
                self._switch(PHASE_TRANSITION)
                self.trans_count = self.w_trans
            elif self.no_progress >= self.w_stag:
                self._switch(PHASE_EXPLORATION)

        # ── EXPLORATION: watch for spike or time-out ──────────────
        elif self.current_phase == PHASE_EXPLORATION:
            if progress_delta >= self.spike:
                self._switch(PHASE_TRANSITION)
                self.trans_count = self.w_trans
            elif self.phase_step >= self.w_explore:
                self._switch(PHASE_EXECUTION)

        phase_changed = (self.current_phase != prev_phase)
        return self.current_phase, phase_changed

    def get_phase_stats(self):
        """Return dict of per-phase statistics for state construction."""
        return {
            "phase_type"         : self.current_phase,
            "phase_progress_gain": self.phase_progress_gain,
            "phase_inertia_ratio": self.phase_inertia_calls / max(self.phase_step, 1),
            "phase_length"       : self.phase_step,
            "phase_token_total"  : self.phase_token_total,
        }


# ────────────────────────────────────────────────────────────────────
# Q-Network  (same architecture as DQN baseline for fair comparison)
# ────────────────────────────────────────────────────────────────────
class SimpleQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


# ────────────────────────────────────────────────────────────────────
# HierarchicalRLAgent  (Semi-MDP DQN)
# ────────────────────────────────────────────────────────────────────
class HierarchicalRLAgent:
    """Semi-MDP DQN macro-controller.

    Fires at phase boundaries (Exploration ↔ Execution ↔ Transition).
    State: 7-dim summary of the *completed* phase.
    Reward: phase-level efficiency signal.
    """

    def __init__(self,
                 state_dim=7,
                 action_dim=3,
                 lr=0.001,
                 gamma=0.9,
                 epsilon=0.3,
                 epsilon_min=0.05,
                 epsilon_decay=0.9990,
                 threshold_map=None,
                 transition_threshold=None):

        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Configurable threshold maps (fall back to module-level defaults)
        self.threshold_map       = threshold_map or dict(THRESHOLD_MAP)
        self.transition_threshold = transition_threshold if transition_threshold is not None else TRANSITION_THRESHOLD

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net   = SimpleQNet(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory  = deque(maxlen=5000)
        self.batch_size = 64

        # Current macro-action state
        self.current_action    = 1     # default: Balanced
        self.current_threshold = self.threshold_map[1]
        self._phase_start_state  = None
        self._phase_start_action = None

        # ── Structured logging for paper figures ──
        project_path = os.environ.get("PROJECT_PATH", os.getcwd())
        self._data_dir = os.environ.get(
            "HIER_DATA_DIR",
            os.path.join(project_path, "experiment_data"))
        self._rl_log = []       # (episode, loss, reward, epsilon, q_mean)
        self._learn_step = 0

        print("🏛️  [HierInertia] HierarchicalRLAgent initialized (Semi-MDP, state_dim=7).",
              flush=True)

    # ──────────────────────────────────────────────────────────────
    # State construction
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def build_state(phase_type, phase_progress_gain, phase_inertia_ratio,
                    phase_length, cumulative_progress,
                    remaining_steps, max_steps):
        """Construct 7-dim normalized state vector."""
        progress_efficiency = (phase_progress_gain / max(phase_length, 1))
        state = np.array([
            phase_type         / 2.0,                          # 0-1
            min(phase_progress_gain, 1.0),                      # 0-1
            min(phase_inertia_ratio, 1.0),                      # 0-1
            min(phase_length   / max(max_steps, 1), 1.0),       # 0-1
            min(progress_efficiency * 10.0, 1.0),               # 0-1  (efficiency)
            max(remaining_steps / max(max_steps, 1), 0.0),      # 0-1
            min(cumulative_progress, 1.0),                      # 0-1
        ], dtype=np.float32)
        return state

    # ──────────────────────────────────────────────────────────────
    # Phase boundary callbacks
    # ──────────────────────────────────────────────────────────────
    def on_phase_start(self, state):
        """Called when a new phase begins. Returns new inertia threshold."""
        self._phase_start_state  = state.copy()
        self._phase_start_action = self.choose_action(state)
        self.current_action      = self._phase_start_action
        self.current_threshold   = self.threshold_map.get(self.current_action, 0.3)
        return self.current_threshold

    def on_phase_end(self, next_state, phase_reward, done):
        """Called when a phase ends. Stores transition and triggers learning."""
        if self._phase_start_state is not None:
            self.store_transition(
                self._phase_start_state,
                self._phase_start_action,
                phase_reward,
                next_state,
                done,
            )
            self.learn()

    # ──────────────────────────────────────────────────────────────
    # Standard DQN methods
    # ──────────────────────────────────────────────────────────────
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.q_net(state_t)).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state      = torch.FloatTensor(np.array(state)).to(self.device)
        action     = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        q_eval = self.q_net(state).gather(1, action)
        with torch.no_grad():
            q_next   = self.q_net(next_state).max(1)[0].view(self.batch_size, 1)
            q_target = reward + self.gamma * q_next * (1 - done)

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log RL metrics
        self._learn_step += 1
        self._rl_log.append({
            "step": self._learn_step,
            "loss": float(loss.item()),
            "q_mean": float(q_eval.mean().item()),
            "q_std": float(q_eval.std().item()),
            "epsilon": float(self.epsilon),
            "memory_size": len(self.memory),
        })

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_rl_curve(self, tag="default"):
        """Save RL learning curve to experiment_data/rl_curves/."""
        if not self._rl_log:
            return
        out_dir = os.path.join(self._data_dir, "rl_curves")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"rl_curve_{tag}.jsonl")
        with open(path, "w") as f:
            for rec in self._rl_log:
                f.write(json.dumps(rec) + "\n")
        print(f"[DATA] RL curve saved → {path} ({len(self._rl_log)} entries)")


# ────────────────────────────────────────────────────────────────────
# EpisodeTracker — structured per-step data recorder for paper figures
# ────────────────────────────────────────────────────────────────────
class EpisodeTracker:
    """Records step-level and phase-level data for one episode.

    Produces 4 types of data used for paper figures / tables:
      1. step_log   → progress curve, per-step inertia decision
      2. phase_log  → phase timeline, phase distribution
      3. inertia_log → inertia trigger/success breakdown
      4. episode_summary → per-episode aggregates
    """

    def __init__(self, episode_index, task_name="", max_steps=30):
        self.episode_index = episode_index
        self.task_name = task_name
        self.max_steps = max_steps

        # Per-step records
        self.step_log = []
        # Per-phase records
        self.phase_log = []
        # Phase accumulator
        self._phase_start_step = 0
        self._phase_start_progress = 0.0

    def log_step(self, step, action, progress, progress_delta,
                 phase, threshold, inertia_used, inertia_success,
                 cips_score=0.0, step_tokens=0, dqn_action=None):
        """Record one step."""
        self.step_log.append({
            "step": step,
            "action": action,
            "progress": round(progress, 4),
            "progress_delta": round(progress_delta, 4),
            "phase": int(phase),                     # 0/1/2
            "threshold": round(threshold, 3),
            "inertia_used": bool(inertia_used),
            "inertia_success": bool(inertia_success),  # inertia used AND progress went up
            "cips_score": round(cips_score, 4),
            "step_tokens": step_tokens,
            "dqn_action": dqn_action,                # None if DQN not fired this step
        })

    def log_phase_end(self, step, phase_type, phase_reward,
                      progress_at_end, dqn_action, threshold):
        """Record one completed phase."""
        length = step - self._phase_start_step
        progress_gain = progress_at_end - self._phase_start_progress
        # Count inertia usage within this phase
        phase_steps = [s for s in self.step_log
                       if self._phase_start_step <= s["step"] < step]
        inertia_count = sum(1 for s in phase_steps if s["inertia_used"])
        inertia_ok = sum(1 for s in phase_steps if s["inertia_success"])
        phase_tokens = sum(s["step_tokens"] for s in phase_steps)

        self.phase_log.append({
            "phase_type": int(phase_type),
            "start_step": self._phase_start_step,
            "end_step": step,
            "length": length,
            "progress_gain": round(progress_gain, 4),
            "inertia_count": inertia_count,
            "inertia_success": inertia_ok,
            "inertia_ratio": round(inertia_count / max(length, 1), 3),
            "phase_reward": round(phase_reward, 4) if phase_reward is not None else None,
            "dqn_action": dqn_action,
            "threshold": round(threshold, 3),
            "phase_tokens": phase_tokens,
        })
        self._phase_start_step = step
        self._phase_start_progress = progress_at_end

    def get_episode_summary(self, done, final_progress, total_tokens, total_calls):
        """Aggregate episode-level metrics."""
        phase_counts = {0: 0, 1: 0, 2: 0}
        phase_steps_total = {0: 0, 1: 0, 2: 0}
        for p in self.phase_log:
            phase_counts[p["phase_type"]] += 1
            phase_steps_total[p["phase_type"]] += p["length"]
        total_steps = len(self.step_log)
        total_inertia = sum(1 for s in self.step_log if s["inertia_used"])
        total_inertia_ok = sum(1 for s in self.step_log if s["inertia_success"])

        return {
            "episode": self.episode_index,
            "task_name": self.task_name,
            "done": done,
            "final_progress": round(final_progress, 4),
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "inertia_total": total_inertia,
            "inertia_success": total_inertia_ok,
            "inertia_rate": round(total_inertia / max(total_steps, 1), 3),
            "inertia_precision": round(total_inertia_ok / max(total_inertia, 1), 3),
            "num_phases": len(self.phase_log),
            "phase_counts": phase_counts,          # {0: N, 1: N, 2: N}
            "phase_steps": phase_steps_total,      # steps spent in each phase
            "num_phase_changes": len(self.phase_log) - 1 if self.phase_log else 0,
        }

    def to_dict(self):
        return {
            "step_log": self.step_log,
            "phase_log": self.phase_log,
        }


class ExperimentDataSaver:
    """Collects EpisodeTrackers and saves structured data for paper figures."""

    def __init__(self, experiment_tag, data_dir=None):
        self.tag = experiment_tag
        if data_dir is None:
            project_path = os.environ.get("PROJECT_PATH", os.getcwd())
            data_dir = os.path.join(project_path, "experiment_data")
        self.data_dir = data_dir
        self.episodes = []         # list of EpisodeTracker.to_dict()
        self.summaries = []        # list of episode summaries

    def add_episode(self, tracker, done, final_progress, total_tokens, total_calls):
        summary = tracker.get_episode_summary(done, final_progress, total_tokens, total_calls)
        self.summaries.append(summary)
        self.episodes.append(tracker.to_dict())

    def save_all(self):
        """Write all data files."""
        os.makedirs(self.data_dir, exist_ok=True)

        # 1. Phase logs — all phase transitions across all episodes
        phase_dir = os.path.join(self.data_dir, "phase_logs")
        os.makedirs(phase_dir, exist_ok=True)
        path = os.path.join(phase_dir, f"phases_{self.tag}.jsonl")
        with open(path, "w") as f:
            for ep_idx, ep in enumerate(self.episodes):
                for p in ep["phase_log"]:
                    p["episode"] = ep_idx
                    f.write(json.dumps(p) + "\n")
        print(f"[DATA] Phase logs → {path}")

        # 2. Progress curves — step-level progress for every episode
        prog_dir = os.path.join(self.data_dir, "progress_curves")
        os.makedirs(prog_dir, exist_ok=True)
        path = os.path.join(prog_dir, f"progress_{self.tag}.jsonl")
        with open(path, "w") as f:
            for ep_idx, ep in enumerate(self.episodes):
                for s in ep["step_log"]:
                    s["episode"] = ep_idx
                    f.write(json.dumps(s) + "\n")
        print(f"[DATA] Progress curves → {path}")

        # 3. Inertia logs — per-episode inertia usage
        iner_dir = os.path.join(self.data_dir, "inertia_logs")
        os.makedirs(iner_dir, exist_ok=True)
        path = os.path.join(iner_dir, f"inertia_{self.tag}.jsonl")
        with open(path, "w") as f:
            for s in self.summaries:
                f.write(json.dumps(s) + "\n")
        print(f"[DATA] Inertia summaries → {path}")

        # 4. Summary table
        sum_dir = os.path.join(self.data_dir, "summary")
        os.makedirs(sum_dir, exist_ok=True)
        path = os.path.join(sum_dir, f"summary_{self.tag}.json")
        agg = self._aggregate()
        with open(path, "w") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"[DATA] Summary → {path}")

    def _aggregate(self):
        """Compute aggregate statistics for the experiment."""
        n = len(self.summaries)
        if n == 0:
            return {}
        sr = sum(1 for s in self.summaries if s["done"]) / n
        pr = sum(s["final_progress"] for s in self.summaries) / n
        avg_steps = sum(s["total_steps"] for s in self.summaries) / n
        avg_tokens = sum(s["total_tokens"] for s in self.summaries) / n
        avg_inertia_rate = sum(s["inertia_rate"] for s in self.summaries) / n
        avg_inertia_prec = sum(s["inertia_precision"] for s in self.summaries) / n
        # Phase distribution
        total_phase_steps = {0: 0, 1: 0, 2: 0}
        total_phase_count = {0: 0, 1: 0, 2: 0}
        for s in self.summaries:
            for k in [0, 1, 2]:
                total_phase_steps[k] += s["phase_steps"].get(str(k), s["phase_steps"].get(k, 0))
                total_phase_count[k] += s["phase_counts"].get(str(k), s["phase_counts"].get(k, 0))
        all_steps = sum(total_phase_steps.values()) or 1
        return {
            "experiment": self.tag,
            "num_episodes": n,
            "SR": round(sr, 4),
            "PR": round(pr, 4),
            "avg_steps": round(avg_steps, 1),
            "avg_tokens": round(avg_tokens, 0),
            "avg_inertia_rate": round(avg_inertia_rate, 3),
            "avg_inertia_precision": round(avg_inertia_prec, 3),
            "phase_distribution": {
                "exploration_pct": round(total_phase_steps[0] / all_steps, 3),
                "execution_pct": round(total_phase_steps[1] / all_steps, 3),
                "transition_pct": round(total_phase_steps[2] / all_steps, 3),
            },
            "phase_counts": {
                "exploration": total_phase_count[0],
                "execution": total_phase_count[1],
                "transition": total_phase_count[2],
            },
        }
