#!/usr/bin/env python3
"""Plot recorded losses without modifying the frozen training source or exports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.artifacts
    paths = {
        'Baseline': base / 'runs/baseline_20m_fineweb_edu_50m_v1/metrics.jsonl',
        'Midpoint': base / 'reversible-v1/runs/midpoint_20m_fineweb_edu_50m_v1_matched/metrics.jsonl',
        'Euler': base / 'reversible-v1/runs/euler_20m_fineweb_edu_50m_v1_matched/metrics.jsonl',
    }
    colors = {'Baseline': '#54616c', 'Midpoint': '#007b9a', 'Euler': '#b14d45'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for label, path in paths.items():
        events = [json.loads(line) for line in path.read_text().splitlines()]
        for axis, kind, field in zip(axes, ['training_progress', 'validation'],
                                    ['training_loss_this_update', 'validation_loss']):
            rows = [e for e in events if e['event'] == kind]
            axis.plot([e['committed_targets'] / 1e6 for e in rows],
                      [e[field] for e in rows], label=label, color=colors[label],
                      linewidth=1.2, marker='o' if kind == 'validation' else None,
                      markersize=4, alpha=1 if kind == 'validation' else .8)
    axes[0].set_title('Training window loss (169 sampled updates)')
    axes[1].set_title('Validation loss (1M held-out targets)')
    axes[1].axhline(5.562294847167969, color='#555555', linestyle=':', linewidth=1,
                   label='Final-loss acceptance cutoff')
    for axis in axes:
        axis.set_xlabel('Committed training targets (millions)')
        axis.set_ylabel('Loss (nats / valid target)')
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle('20.34M parameters · T4 FP16 · physical batch 29 · effective batch 58', fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
