"""
RL Dynamic Learning Curve Extractor & Plotter
Usage: python plot_rl_curve.py <log_path> [output_csv] [output_png]

Extracts per-episode metrics from agent run.log and plots:
  - X: Episode index (RL training timeline)
  - Y1 (left): Per-episode PR + rolling mean PR (window=10)
  - Y2 (right): A1 activation rate (inertia policy confidence)
"""

import re
import sys
import json
import os
import csv

def extract_per_episode(log_path):
    """Parse [TOKEN], [TIMER], [INERTIA] records per episode from run.log"""
    episodes = {}

    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[TOKEN\] Example (\d+): in=(\d+), out=(\d+), calls=(\d+)', line)
            if m:
                eid = int(m.group(1))
                episodes.setdefault(eid, {})
                episodes[eid]['tok_in'] = int(m.group(2))
                episodes[eid]['tok_out'] = int(m.group(3))
                episodes[eid]['llm_calls'] = int(m.group(4))
                continue

            m = re.search(r'\[TIMER\] Example (\d+): elapsed=([\d.]+)s', line)
            if m:
                eid = int(m.group(1))
                episodes.setdefault(eid, {})
                episodes[eid]['elapsed'] = float(m.group(2))
                continue

            m = re.search(r'\[INERTIA\] Example (\d+): steps=(\d+), A0=(\d+), A1=(\d+), A2=(\d+)', line)
            if m:
                eid = int(m.group(1))
                episodes.setdefault(eid, {})
                episodes[eid]['steps'] = int(m.group(2))
                episodes[eid]['A0'] = int(m.group(3))
                episodes[eid]['A1'] = int(m.group(4))
                episodes[eid]['A2'] = int(m.group(5))
    return episodes

def load_scienceworld_txt(txt_path):
    """Load per-episode SR/PR/GA from scienceworld.txt"""
    results = {}
    with open(txt_path) as f:
        for i, line in enumerate(f):
            m = re.search(r'\[EXP\] (\d+): \[success_rate\]: (\w+), \[progress_rate\]: ([\d.]+), \[grounding_acc\]: ([\d.]+)', line)
            if m:
                eid = int(m.group(1))
                results[eid] = {
                    'sr': 1 if m.group(2) == 'True' else 0,
                    'pr': float(m.group(3)),
                    'ga': float(m.group(4)),
                }
    return results

def load_difficulty(jsonl_path):
    difficulty = {}
    try:
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                difficulty[d['id']] = d.get('difficulty', 'unknown')
    except:
        pass
    return difficulty

def rolling_mean(values, window=10):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result

def main():
    base = '/root/autodl-tmp/AutoTool/results/exp43_sw_qwen_rl_v3_20260304'
    log_path = os.path.join(base, 'run.log')
    txt_path = os.path.join(base, 'scienceworld.txt')
    jsonl_path = '/root/autodl-tmp/AutoTool/data/scienceworld/test.jsonl'
    out_csv = os.path.join(base, 'rl_curve_data.csv')
    out_png = os.path.join(base, 'rl_learning_curve.png')

    if len(sys.argv) > 1: log_path = sys.argv[1]
    if len(sys.argv) > 2: out_csv = sys.argv[2]
    if len(sys.argv) > 3: out_png = sys.argv[3]

    print(f"Loading from: {log_path}")
    episodes_log = extract_per_episode(log_path)
    episodes_txt = load_scienceworld_txt(txt_path)
    difficulty = load_difficulty(jsonl_path)

    # Merge
    all_ids = sorted(set(list(episodes_txt.keys())))
    rows = []
    for eid in all_ids:
        row = {'episode': eid}
        row.update(episodes_txt.get(eid, {'sr': None, 'pr': None, 'ga': None}))
        row.update(episodes_log.get(eid, {}))
        row['difficulty'] = difficulty.get(eid, 'unknown')
        total_act = row.get('A0', 0) + row.get('A1', 0) + row.get('A2', 0)
        row['a1_rate'] = row.get('A1', 0) / total_act if total_act > 0 else 0
        rows.append(row)

    # Compute rolling metrics
    pr_list = [r.get('pr', 0) or 0 for r in rows]
    sr_list = [r.get('sr', 0) or 0 for r in rows]
    a1_list = [r.get('a1_rate', 0) for r in rows]
    roll_pr = rolling_mean(pr_list, window=10)
    roll_sr = rolling_mean(sr_list, window=10)
    cumul_sr = [sum(sr_list[:i+1])/(i+1)*100 for i in range(len(sr_list))]

    for i, row in enumerate(rows):
        row['roll_pr'] = roll_pr[i]
        row['roll_sr'] = roll_sr[i]
        row['cumul_sr'] = cumul_sr[i]

    # Write CSV
    fieldnames = ['episode', 'difficulty', 'sr', 'pr', 'ga', 'roll_pr', 'roll_sr',
                  'cumul_sr', 'a1_rate', 'A0', 'A1', 'A2', 'steps',
                  'llm_calls', 'tok_in', 'tok_out', 'elapsed']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {out_csv}  ({len(rows)} episodes)")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        episodes_x = [r['episode'] for r in rows]
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        # --- Top panel: PR + Rolling PR + Cumulative SR ---
        ax1.scatter(episodes_x, [r.get('pr',0) for r in rows],
                    alpha=0.25, s=18, c='steelblue', label='Per-episode PR', zorder=2)
        ax1.plot(episodes_x, roll_pr, color='steelblue', linewidth=2,
                 label='Rolling PR (w=10)', zorder=3)

        ax2 = ax1.twinx()
        ax2.plot(episodes_x, cumul_sr, color='darkorange', linewidth=1.8,
                 linestyle='--', label='Cumulative SR (%)', zorder=3)

        # Difficulty shading
        for r in rows:
            if r.get('difficulty') == 'easy':
                ax1.axvspan(r['episode']-0.5, r['episode']+0.5, alpha=0.07,
                            color='green', zorder=0)

        ax1.set_ylabel('Progress Rate', fontsize=12)
        ax1.set_ylim(0, 1.05)
        ax2.set_ylabel('Cumulative SR (%)', fontsize=12, color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax1.set_title('Qwen2.5-72B · ScienceWorld · RL v3 (θ=0.2, mem=25)\nOnline RL Learning Curve',
                      fontsize=13)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        ax1.grid(alpha=0.3)

        # --- Bottom panel: A1 activation rate ---
        ax3.bar(episodes_x, a1_list, color='coral', alpha=0.7, label='A1 rate (Inertia / Total)')
        roll_a1 = rolling_mean(a1_list, window=10)
        ax3.plot(episodes_x, roll_a1, color='darkred', linewidth=2, label='Rolling A1 rate (w=10)')
        ax3.set_xlabel('Episode Index (RL Timeline)', fontsize=12)
        ax3.set_ylabel('Inertia Activation Rate', fontsize=12)
        ax3.set_ylim(0, 1.0)
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(alpha=0.3)

        # Add text annotations
        n = len(rows)
        final_sr = cumul_sr[-1] if cumul_sr else 0
        avg_a1 = sum(a1_list)/len(a1_list) if a1_list else 0
        fig.text(0.99, 0.02,
                 f'N={n}  Final SR={final_sr:.1f}%  Avg A1={avg_a1*100:.1f}%',
                 ha='right', fontsize=9, color='gray')

        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {out_png}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot (CSV data is ready)")
    except Exception as e:
        print(f"Plot error: {e} (CSV data is still ready)")

if __name__ == '__main__':
    main()
