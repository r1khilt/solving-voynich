#!/usr/bin/env python3
"""Render a descriptive TEACH-0010 summary from the audited report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "TEACH-0010"
COLORS = ("#2457C5", "#E05A47")


def main():
    report = json.loads((RESULT_DIR / "report.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    width = .36

    masks = (7, 11, 13, 14, 15)
    labels = ("0+1+2", "0+1+3", "0+2+3", "1+2+3", "All")
    x = np.arange(len(labels))
    for rep, color, offset in zip(("0", "1"), COLORS, (-width / 2, width / 2)):
        values = [100 * report["discovery_screen"][rep][str(mask)]["donor_items"][
            "accuracy"] for mask in masks]
        axes[0].bar(x + offset, values, width, color=color, label=f"seed {rep}")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_title("No smaller subset is robust across seeds", loc="left", fontweight="bold")
    axes[0].set_ylabel("discovery donor answers (%)")
    axes[0].legend(frameon=False)

    conditions = ("selected_heads", "post_attention", "mlp_only", "random",
                  "cyclic_recipient", "fixed_g0")
    labels = ("All heads", "Post-attn", "MLP", "Random", "Wrong G", "Repeat G0")
    x = np.arange(len(labels))
    for rep, color, offset in zip(("0", "1"), COLORS, (-width / 2, width / 2)):
        values = [100 * report["metrics"][rep]["scores"][name]["donor_items"][
            "accuracy"] for name in conditions]
        axes[1].bar(x + offset, values, width, color=color)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_title("The answer write is recipient-specific", loc="left", fontweight="bold")
    axes[1].set_ylabel("confirmation donor answers (%)")

    for axis in axes:
        axis.set_ylim(0, 105)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9DEE8", linewidth=.8, alpha=.75)
        axis.set_axisbelow(True)
    fig.suptitle("TEACH-0010 · Distributed downstream object write",
                 fontsize=16, fontweight="bold")
    fig.savefig(RESULT_DIR / "answer-write-summary.png", dpi=180, facecolor="white")
    fig.savefig(RESULT_DIR / "answer-write-summary.svg", facecolor="white")


if __name__ == "__main__":
    main()
