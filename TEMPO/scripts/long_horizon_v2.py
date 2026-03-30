"""
Long-Horizon Token Efficiency Analysis (v2)
Compares ReAct, AutoTool, HierInertia on:
  1. PR@N and SR@N curves (step budget analysis)
  2. Token-SR efficiency scatter
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE   = '/root/autodl-tmp/AutoTool/results'
OUT    = os.path.join(BASE, 'figures')
LOGDIR = '/root/autodl-tmp/AutoTool/logs'
os.makedirs(OUT, exist_ok=True)

# ── Experiment config ─────────────────────────────────────────────────────────
EXPS = {
    'ReAct':       'sw_react_baseline_qwen_20260223',
    'AutoTool':    'exp86_sw_qwen_autotool_t01_mem100',
    'HierInertia': 'exp_h4_sw_qwen_hier',
}
COLORS  = {'ReAct':'#2196F3', 'AutoTool':'#FF9800', 'HierInertia':'#F44336'}
MARKERS = {'ReAct':'o',       'AutoTool':'s',        'HierInertia':'D'}
LINES   = {'ReAct':'-',       'AutoTool':'--',       'HierInertia':'-'}

# Avg tokens per episode (empirically measured from logs)
# ReAct:        scienceworld_react_baseline_log_20260301_130228.json  -> 55244
# AutoTool:     exp86.log  -> 37990/ep
# HierInertia:  experiment_data/summary/summary_exp_h4_sw_qwen_hier.json -> 48058
AVG_TOKENS = {
    'ReAct':       55244,
    'AutoTool':    37990,
    'HierInertia': 48058,
}

MAX_STEPS = 30


def read_jsonl_multi(path):
    items = []
    with open(path) as f:
        decoder = json.JSONDecoder()
        buf = f.read()
        pos = 0
        while pos < len(buf):
            s = buf[pos:]; ss = s.lstrip()
            if not ss: break
            pos += len(s) - len(ss)
            try:
                obj, sz = decoder.raw_decode(ss)
                items.append(obj); pos += sz
            except: pos += 1
    return items


def get_pr_at_step(ep, n):
    traj_len = len(ep.get('trajectory', {}))
    if ep.get('is_done') and (traj_len - 1) <= n:
        return ep.get('progress_rate', 0.0)
    sc = ep.get('score_change_record', ep.get('score_state', []))
    pr = 0.0
    for entry in sc:
        if entry[0] <= n:
            pr = max(pr, entry[1])
        else:
            break
    return pr


def compute_curves(episodes):
    n = len(episodes)
    sr, pr = [], []
    for step in range(MAX_STEPS + 1):
        sr.append(sum(1 for ep in episodes
                      if ep.get('is_done') and len(ep.get('trajectory',{}))-1 <= step) / n)
        pr.append(np.mean([get_pr_at_step(ep, step) for ep in episodes]))
    return sr, pr


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading...")
data = {}
for name, dirname in EXPS.items():
    eps = read_jsonl_multi(os.path.join(BASE, dirname, 'logs', 'scienceworld.jsonl'))
    sr, pr = compute_curves(eps)
    data[name] = dict(sr=sr, pr=pr, final_sr=sr[MAX_STEPS], final_pr=pr[MAX_STEPS], eps=eps)
    print(f"  {name}: SR={sr[MAX_STEPS]:.4f}  PR={pr[MAX_STEPS]:.4f}  avg_tok={AVG_TOKENS[name]:,}")

steps = list(range(MAX_STEPS + 1))

# ── Figure: 2 subplots side by side ──────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5))
gs  = gridspec.GridSpec(1, 2, wspace=0.35)

# ── Left: PR@N curves ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

for name in EXPS:
    d = data[name]
    label = f"{name}  (PR={d['final_pr']*100:.1f}%)"
    ax1.plot(steps, [v*100 for v in d['pr']],
             color=COLORS[name], linestyle=LINES[name],
             marker=MARKERS[name], markevery=5, linewidth=2.2, markersize=7,
             label=label)

# Annotate the growing gap between HierInertia and AutoTool
gap_steps = [5, 10, 15, 20, 25, 30]
for s in [15, 25]:
    hi = data['HierInertia']['pr'][s]*100
    at = data['AutoTool']['pr'][s]*100
    ax1.annotate('', xy=(s, hi), xytext=(s, at),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax1.text(s + 0.4, (hi+at)/2, f'+{hi-at:.1f}pp', fontsize=8.5,
             color='gray', va='center')

ax1.set_xlabel('Step Budget', fontsize=12)
ax1.set_ylabel('Average Progress Rate (%)', fontsize=12)
ax1.set_title('(a) PR vs Step Budget\n(ScienceWorld, Qwen2.5-72B)', fontsize=12)
ax1.set_xlim(0, MAX_STEPS); ax1.set_ylim(0, 100)
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=10)

# ── Right: Token-SR efficiency scatter ────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

tok_k = {n: AVG_TOKENS[n]/1000 for n in EXPS}
sr_vals = {n: data[n]['final_sr']*100 for n in EXPS}
pr_vals  = {n: data[n]['final_pr']*100 for n in EXPS}

# Draw "efficiency iso-line" (SR/token = const) through ReAct point
x_range = np.linspace(30, 65, 100)
for iso_label, iso_color, iso_name in [
        ('ReAct iso-line\n(SR / token = const)', '#2196F3', 'ReAct'),
        ('AutoTool iso-line', '#FF9800', 'AutoTool')]:
    pass  # skip iso-lines, too cluttered

# Draw convex hull / Pareto frontier arrow
ax2.annotate('', xy=(tok_k['HierInertia'], sr_vals['HierInertia']),
             xytext=(tok_k['AutoTool'], sr_vals['AutoTool']),
             arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=2, linestyle='dashed'))
ax2.text(43, 59.5, 'HierInertia\nrecovery', fontsize=9, color='#9C27B0', ha='center')

# Scatter points
for name in EXPS:
    ax2.scatter(tok_k[name], sr_vals[name], color=COLORS[name],
                marker=MARKERS[name], s=180, zorder=5,
                label=f"{name}\n({tok_k[name]:.0f}K tok, SR={sr_vals[name]:.1f}%)")
    offset = {'ReAct': (1, 1), 'AutoTool': (-7, -4), 'HierInertia': (1, 1)}[name]
    ax2.annotate(name,
                 xy=(tok_k[name], sr_vals[name]),
                 xytext=(tok_k[name]+offset[0], sr_vals[name]+offset[1]),
                 fontsize=10, color=COLORS[name], fontweight='bold')

# Draw "token saved vs ReAct" annotation
react_tok = tok_k['ReAct']
for name in ['AutoTool', 'HierInertia']:
    saved = (react_tok - tok_k[name]) / react_tok * 100
    sr_loss = sr_vals['ReAct'] - sr_vals[name]
    ax2.annotate(f'−{saved:.0f}% tok\n−{sr_loss:.1f}pp SR',
                 xy=(tok_k[name], sr_vals[name]),
                 xytext=(tok_k[name] - 5, sr_vals[name] - 8),
                 fontsize=8, color=COLORS[name],
                 arrowprops=dict(arrowstyle='->', color=COLORS[name], lw=1))

ax2.set_xlabel('Avg Token Cost per Episode (K tokens)', fontsize=12)
ax2.set_ylabel('Success Rate (%)', fontsize=12)
ax2.set_title('(b) Token-SR Efficiency Tradeoff\n(ScienceWorld, Qwen2.5-72B)', fontsize=12)
ax2.set_xlim(28, 65); ax2.set_ylim(40, 90)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=10)

# Add "ideal" direction arrow
ax2.annotate('Ideal direction', xy=(33, 84), xytext=(33, 75),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=9, color='green', ha='center')
ax2.text(33, 73, '(fewer tokens,\nhigher SR)', fontsize=8, color='green', ha='center')

plt.suptitle('Long-Horizon Token Efficiency Analysis — ScienceWorld (Qwen2.5-72B)',
             fontsize=13, fontweight='bold', y=1.02)

out = os.path.join(OUT, 'token_efficiency_analysis.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close()

# ── SR@N figure (clean, separate) ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for name in EXPS:
    d = data[name]
    ax.plot(steps, [v*100 for v in d['sr']],
            color=COLORS[name], linestyle=LINES[name],
            marker=MARKERS[name], markevery=5, linewidth=2.2, markersize=7,
            label=f"{name}  (SR={d['final_sr']*100:.1f}%)")
ax.set_xlabel('Step Budget', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('Success Rate vs Step Budget\n(ScienceWorld, Qwen2.5-72B)', fontsize=13)
ax.set_xlim(0, MAX_STEPS); ax.set_ylim(0, 100)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'sr_vs_steps_v2.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Method':<14} {'SR':>7} {'PR':>7} {'avg_tok':>9} {'ΔSR vs R':>10} {'Δtok vs R':>11}")
print("-"*65)
react_sr  = data['ReAct']['final_sr']
react_pr  = data['ReAct']['final_pr']
react_tok = AVG_TOKENS['ReAct']
for name in EXPS:
    d = data[name]
    dsr  = (d['final_sr'] - react_sr) * 100
    dtok = (AVG_TOKENS[name] - react_tok) / react_tok * 100
    print(f"{name:<14} {d['final_sr']*100:>6.1f}% {d['final_pr']*100:>6.1f}% "
          f"{AVG_TOKENS[name]:>8,} {dsr:>+9.1f}pp {dtok:>+10.1f}%")

print()
print("PR@N gap (HierInertia − AutoTool):")
for s in [10, 15, 20, 25, 30]:
    gap = (data['HierInertia']['pr'][s] - data['AutoTool']['pr'][s])*100
    print(f"  @step {s}: +{gap:.1f}pp")
