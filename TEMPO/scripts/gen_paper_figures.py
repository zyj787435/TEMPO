"""
Paper figures: GPT-3.5-Turbo ScienceWorld comparison
ReAct vs AutoTool vs HierInertia (v2)

Three metrics: Success Rate (SR), Progress Rate (PR), Token Consumption
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Data ──────────────────────────────────────────────────────────────────────
# GPT-3.5-Turbo ScienceWorld (90 episodes)
methods = ['ReAct', 'AutoTool', 'HierInertia\n(Ours)']
methods_short = ['ReAct', 'AutoTool', 'HierInertia']

SR  = [0.6444, 0.5111, 0.6667]
PR  = [0.7272, 0.5939, 0.7592]
TOK = [82494,  54048,  48047]   # avg tokens/episode

# Colors: neutral, neutral, highlight
COLORS = ['#5B9BD5', '#ED7D31', '#70AD47']
HIGHLIGHT = '#70AD47'
EDGE = 'white'
BAR_WIDTH = 0.32

OUT_DIR = '/root/autodl-tmp/AutoTool/results/paper_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Figure 1: Main comparison (SR + PR side by side, tokens below) ────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
fig.patch.set_facecolor('white')

x = np.arange(len(methods))

# ── Panel (a): Success Rate ──
ax = axes[0]
bars = ax.bar(x, SR, color=COLORS, edgecolor=EDGE, linewidth=0.5, zorder=3, width=0.55)
ax.set_ylim(0, 0.85)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Success Rate', fontsize=11)
ax.set_title('(a) Success Rate (SR)', fontsize=11, pad=8)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

# Value labels
for bar, val in zip(bars, SR):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# Annotate HierInertia improvement over ReAct
delta_sr = SR[2] - SR[0]
ax.annotate('', xy=(x[2], SR[2]+0.035), xytext=(x[0], SR[2]+0.035),
            arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.2))
ax.text((x[0]+x[2])/2, SR[2]+0.042, f'+{delta_sr*100:.1f}pp',
        ha='center', va='bottom', fontsize=8.5, color='#333333')

# ── Panel (b): Progress Rate ──
ax = axes[1]
bars = ax.bar(x, PR, color=COLORS, edgecolor=EDGE, linewidth=0.5, zorder=3, width=0.55)
ax.set_ylim(0, 0.95)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Progress Rate', fontsize=11)
ax.set_title('(b) Progress Rate (PR)', fontsize=11, pad=8)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

for bar, val in zip(bars, PR):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

delta_pr = PR[2] - PR[0]
ax.annotate('', xy=(x[2], PR[2]+0.035), xytext=(x[0], PR[2]+0.035),
            arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.2))
ax.text((x[0]+x[2])/2, PR[2]+0.042, f'+{delta_pr*100:.1f}pp',
        ha='center', va='bottom', fontsize=8.5, color='#333333')

# ── Panel (c): Token Consumption ──
ax = axes[2]
TOK_K = [t / 1000 for t in TOK]
bars = ax.bar(x, TOK_K, color=COLORS, edgecolor=EDGE, linewidth=0.5, zorder=3, width=0.55)
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Avg Tokens / Episode (K)', fontsize=11)
ax.set_title('(c) Token Consumption', fontsize=11, pad=8)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

for bar, val in zip(bars, TOK_K):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
            f'{val:.1f}K', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# Annotate savings vs ReAct
saving_pct = (TOK[0] - TOK[2]) / TOK[0] * 100
ax.annotate('', xy=(x[0], TOK_K[2]+4), xytext=(x[0], TOK_K[0]-0.5),
            arrowprops=dict(arrowstyle='<->', color='#C00000', lw=1.2))
ax.text(x[0]+0.32, (TOK_K[0]+TOK_K[2])/2, f'−{saving_pct:.0f}%',
        ha='left', va='center', fontsize=8.5, color='#C00000', fontweight='bold')

plt.suptitle('GPT-3.5-Turbo on ScienceWorld (n=90)', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'gpt35_sw_main_comparison.pdf')
plt.savefig(out_path, bbox_inches='tight', dpi=200)
plt.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
print(f"Saved: {out_path}")
plt.close()


# ── Figure 2: Efficiency frontier (PR vs Tokens scatter) ─────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))
fig.patch.set_facecolor('white')

for i, (m, sr, pr, tok) in enumerate(zip(methods_short, SR, PR, TOK)):
    ax.scatter(tok/1000, pr, s=160, color=COLORS[i], zorder=5,
               edgecolors='white', linewidths=1.5, label=m)
    offsets = {'ReAct': (6, 6), 'AutoTool': (-16, -16), 'HierInertia': (-52, 6)}
    offset = offsets.get(m, (6, 6))
    ax.annotate(m, (tok/1000, pr), xytext=offset, textcoords='offset points',
                fontsize=10, color=COLORS[i], fontweight='bold')

# Draw arrows showing direction of improvement
ax.annotate('Better efficiency\n(↑PR, ↓Tokens)', xy=(52, 0.77), fontsize=9,
            color='gray', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='lightgray'))

ax.set_xlabel('Avg Token Consumption / Episode (K)', fontsize=11)
ax.set_ylabel('Progress Rate (PR)', fontsize=11)
ax.set_title('Efficiency Frontier\nGPT-3.5-Turbo on ScienceWorld', fontsize=11, pad=8)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.xaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(35, 100)
ax.set_ylim(0.53, 0.82)

plt.tight_layout()
out_path2 = os.path.join(OUT_DIR, 'gpt35_sw_efficiency_frontier.pdf')
plt.savefig(out_path2, bbox_inches='tight', dpi=200)
plt.savefig(out_path2.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
print(f"Saved: {out_path2}")
plt.close()


# ── Figure 3: Ablation study (Qwen + GPT side by side) ───────────────────────
# HierInertia full vs AB-1 (flat DQN) vs AB-2 (rule phase) vs AB-3 (no mem trim)
ablation_methods = ['HierInertia\n(Full)', 'AB-1\nFlat DQN', 'AB-2\nRule Phase', 'AB-3\nNo MemTrim']
abl_colors = ['#70AD47', '#9DC3E6', '#FFE699', '#F4B183']

# Qwen2.5-72B data
# Full: exp_h4_sw_qwen_hier (SR=0.6667, PR=0.7354)
SR_abl_qwen = [0.6667, 0.6222, 0.6222, 0.5000]
PR_abl_qwen = [0.7354, 0.7001, 0.7211, 0.6329]

# GPT-4o-mini data
# Full: exp_h5d_sw_gpt_hier_tokbase (SR=0.6778, PR=0.7582)
SR_abl_gpt = [0.6778, 0.6111, 0.6778, 0.6333]
PR_abl_gpt = [0.7582, 0.7162, 0.7742, 0.7078]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor('white')

x_abl = np.arange(len(ablation_methods))
bar_w = 0.38

for ax, SR_d, PR_d, model_name in zip(
        axes,
        [SR_abl_qwen, SR_abl_gpt],
        [PR_abl_qwen, PR_abl_gpt],
        ['Qwen2.5-72B', 'GPT-4o-mini']):

    bars_sr = ax.bar(x_abl - bar_w/2, SR_d, width=bar_w, color=abl_colors,
                     edgecolor='white', linewidth=0.5, zorder=3, label='SR')
    bars_pr = ax.bar(x_abl + bar_w/2, PR_d, width=bar_w, color=abl_colors,
                     edgecolor='white', linewidth=0.5, zorder=3, alpha=0.6, label='PR')

    # Add hatching to PR bars
    for bar in bars_pr:
        bar.set_hatch('///')
        bar.set_edgecolor('white')

    ax.set_ylim(0, 0.88)
    ax.set_xticks(x_abl)
    ax.set_xticklabels(ablation_methods, fontsize=9.5)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Ablation Study — {model_name}', fontsize=11, pad=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    for bars, vals in [(bars_sr, SR_d), (bars_pr, PR_d)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.006,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

# Shared legend
sr_patch = mpatches.Patch(facecolor='gray', label='SR (solid)')
pr_patch = mpatches.Patch(facecolor='gray', alpha=0.5, hatch='///', label='PR (hatched)')
fig.legend(handles=[sr_patch, pr_patch], loc='upper right', fontsize=9,
           bbox_to_anchor=(0.99, 0.99))

plt.suptitle('Ablation Study on ScienceWorld (n=90)', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
out_path3 = os.path.join(OUT_DIR, 'ablation_study_sw.pdf')
plt.savefig(out_path3, bbox_inches='tight', dpi=200)
plt.savefig(out_path3.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
print(f"Saved: {out_path3}")
plt.close()

print("\n✓ All paper figures generated.")
print(f"Output directory: {OUT_DIR}")
