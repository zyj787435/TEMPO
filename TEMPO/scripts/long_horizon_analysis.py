"""
Long-Horizon Performance Analysis for ScienceWorld (Qwen2.5-72B)
Computes SR@N and PR@N curves across step budgets 1..30
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/root/autodl-tmp/AutoTool/results'
OUT_DIR = os.path.join(BASE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

EXPERIMENTS = {
    'ReAct':       'sw_react_baseline_qwen_20260223',
    'AutoTool':    'exp86_sw_qwen_autotool_t01_mem100',
    'AdaInertia':  'rl_sw_v3_20260217_221454',
    'HierInertia': 'exp_h4_sw_qwen_hier',
}

COLORS = {
    'ReAct':       '#2196F3',
    'AutoTool':    '#FF9800',
    'AdaInertia':  '#4CAF50',
    'HierInertia': '#F44336',
}

MARKERS = {
    'ReAct':       'o',
    'AutoTool':    's',
    'AdaInertia':  '^',
    'HierInertia': 'D',
}

MAX_STEPS = 30


def read_jsonl_multi(path):
    items = []
    with open(path) as f:
        decoder = json.JSONDecoder()
        buf = f.read()
        pos = 0
        while pos < len(buf):
            s = buf[pos:]
            s_stripped = s.lstrip()
            if not s_stripped:
                break
            pos += len(s) - len(s_stripped)
            try:
                obj, sz = decoder.raw_decode(s_stripped)
                items.append(obj)
                pos += sz
            except:
                pos += 1
    return items


def get_progress_at_step(episode, n):
    traj_len = len(episode.get('trajectory', {}))
    is_done = episode.get('is_done', False)
    final_pr = episode.get('progress_rate', 0.0)
    if is_done and (traj_len - 1) <= n:
        return final_pr
    sc = episode.get('score_change_record', episode.get('score_state', []))
    pr = 0.0
    for entry in sc:
        step, score = entry[0], entry[1]
        if step <= n:
            pr = max(pr, score)
        else:
            break
    return pr


def compute_curves(episodes, max_steps=MAX_STEPS):
    n_ep = len(episodes)
    sr_curve = []
    pr_curve = []
    for n in range(max_steps + 1):
        successes = sum(
            1 for ep in episodes
            if ep.get('is_done', False) and len(ep.get('trajectory', {})) - 1 <= n
        )
        sr_curve.append(successes / n_ep)
        avg_pr = np.mean([get_progress_at_step(ep, n) for ep in episodes])
        pr_curve.append(float(avg_pr))
    return sr_curve, pr_curve


def avg_steps_to_success(episodes):
    steps = [
        len(ep.get('trajectory', {})) - 1
        for ep in episodes
        if ep.get('is_done', False)
    ]
    return np.mean(steps) if steps else float('nan')


print("Loading experiments...")
data = {}
for name, dirname in EXPERIMENTS.items():
    path = os.path.join(BASE, dirname, 'logs', 'scienceworld.jsonl')
    episodes = read_jsonl_multi(path)
    sr_curve, pr_curve = compute_curves(episodes)
    data[name] = {
        'episodes': episodes,
        'sr_curve': sr_curve,
        'pr_curve': pr_curve,
        'final_sr': sr_curve[MAX_STEPS],
        'final_pr': pr_curve[MAX_STEPS],
        'avg_steps': avg_steps_to_success(episodes),
        'auc_sr': np.trapz(sr_curve),
    }
    print(f"  {name}: n={len(episodes)}, SR={sr_curve[MAX_STEPS]:.4f}, PR={pr_curve[MAX_STEPS]:.4f}")

steps = list(range(MAX_STEPS + 1))

# Plot 1: SR vs Steps
fig, ax = plt.subplots(figsize=(10, 6))
for name in EXPERIMENTS:
    d = data[name]
    label = f"{name} (SR={d['final_sr']*100:.1f}%)"
    ax.plot(steps, [v * 100 for v in d['sr_curve']],
            color=COLORS[name], marker=MARKERS[name], markevery=5,
            linewidth=2, markersize=7, label=label)
ax.set_xlabel('Step Budget', fontsize=13)
ax.set_ylabel('Success Rate (%)', fontsize=13)
ax.set_title('Success Rate vs Step Budget\n(ScienceWorld, Qwen2.5-72B)', fontsize=14)
ax.set_xlim(0, MAX_STEPS)
ax.set_ylim(0, 100)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'sr_vs_steps.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close()

# Plot 2: PR vs Steps
fig, ax = plt.subplots(figsize=(10, 6))
for name in EXPERIMENTS:
    d = data[name]
    label = f"{name} (PR={d['final_pr']*100:.1f}%)"
    ax.plot(steps, [v * 100 for v in d['pr_curve']],
            color=COLORS[name], marker=MARKERS[name], markevery=5,
            linewidth=2, markersize=7, label=label)
ax.set_xlabel('Step Budget', fontsize=13)
ax.set_ylabel('Average Progress Rate (%)', fontsize=13)
ax.set_title('Average Progress Rate vs Step Budget\n(ScienceWorld, Qwen2.5-72B)', fontsize=14)
ax.set_xlim(0, MAX_STEPS)
ax.set_ylim(0, 100)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'pr_vs_steps.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()

# Summary table
print("\n" + "="*72)
print(f"{'Method':<15} {'Final SR':>9} {'Final PR':>9} {'AvgSteps':>10} {'AUC(SR)':>10}")
print("-"*72)
for name in EXPERIMENTS:
    d = data[name]
    print(f"{name:<15} {d['final_sr']*100:>8.1f}% {d['final_pr']*100:>8.1f}% "
          f"{d['avg_steps']:>10.2f} {d['auc_sr']:>10.2f}")

print("\nSR@N at key budgets:")
print(f"{'Method':<15} {'@5':>6} {'@10':>6} {'@15':>6} {'@20':>6} {'@25':>6} {'@30':>6}")
print("-"*55)
for name in EXPERIMENTS:
    sc = data[name]['sr_curve']
    vals_str = "  ".join(f"{sc[n]*100:.0f}%" for n in [5,10,15,20,25,30])
    print(f"{name:<15} {vals_str}")

print("\nPR@N at key budgets:")
print(f"{'Method':<15} {'@5':>6} {'@10':>6} {'@15':>6} {'@20':>6} {'@25':>6} {'@30':>6}")
print("-"*55)
for name in EXPERIMENTS:
    pc = data[name]['pr_curve']
    vals_str = "  ".join(f"{pc[n]*100:.0f}%" for n in [5,10,15,20,25,30])
    print(f"{name:<15} {vals_str}")
