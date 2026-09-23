#!/usr/bin/env python3
"""Render a descriptive TEACH-0008 summary from the audited compact report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "TEACH-0008"


def main():
    report = json.loads((RESULT_DIR / "report.json").read_text())
    colors = ("#2457C5", "#E05A47")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    labels = ["H1", "H3", "H1+H3", "H0+H2", "All"]
    masks = (2, 8, 10, 5, 15)
    x = np.arange(len(labels))
    width = 0.36
    for rep, color, offset in zip(("0", "1"), colors, (-width / 2, width / 2)):
        values = [100 * report["discovery_screen"][rep][str(mask)]["donor_groups"][
            "accuracy"] for mask in masks]
        axes[0].bar(x + offset, values, width, label=f"seed {rep}", color=color)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("exact three-table groups (%)")
    axes[0].set_title("Discovery: heads work together", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    conditions = ("selected_heads", "all_heads", "complement_heads", "mlp_only", "random")
    labels = ("H1+H3", "All heads", "H0+H2", "MLP", "Random")
    x = np.arange(len(labels))
    for rep, color, offset in zip(("0", "1"), colors, (-width / 2, width / 2)):
        values = [100 * report["metrics"][rep]["scores"][condition]["donor_items"][
            "accuracy"] for condition in conditions]
        axes[1].bar(x + offset, values, width, color=color)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("recipient-specific donor answers (%)")
    axes[1].set_title("Confirmation: causal specificity", loc="left", fontweight="bold")

    for rep, color in zip(("0", "1"), colors):
        curve = report["metrics"][rep]["dimension_scores"]
        dims = sorted(int(d) for d in curve)
        values = [100 * curve[str(d)]["donor_items"]["accuracy"] for d in dims]
        axes[2].plot(dims, values, "o-", color=color, linewidth=2.4,
                     markersize=6, label=f"seed {rep}")
    axes[2].set_xticks((1, 2, 4, 8, 11))
    axes[2].set_ylim(0, 105)
    axes[2].set_xlabel("key-subspace dimensions retained")
    axes[2].set_ylabel("donor answers (%)")
    axes[2].set_title("Selected write is low-dimensional", loc="left", fontweight="bold")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9DEE8", linewidth=.8, alpha=.75)
        axis.set_axisbelow(True)
    fig.suptitle("TEACH-0008 · Two attention heads jointly write a reusable key",
                 fontsize=16, fontweight="bold")
    fig.savefig(RESULT_DIR / "head-write-summary.png", dpi=180, facecolor="white")
    fig.savefig(RESULT_DIR / "head-write-summary.svg", facecolor="white")


if __name__ == "__main__":
    main()
