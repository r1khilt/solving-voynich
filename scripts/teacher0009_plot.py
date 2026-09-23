#!/usr/bin/env python3
"""Render a descriptive TEACH-0009 summary from the audited report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "TEACH-0009"
COLORS = ("#2457C5", "#E05A47")


def _item_percent(report, rep, name):
    return 100 * report["metrics"][rep]["scores"][name]["donor_items"]["accuracy"]


def main():
    report = json.loads((RESULT_DIR / "report.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    width = .36

    qkv = ("qkv_bbb", "qkv_dbb", "qkv_bdb", "qkv_bbd", "qkv_ddb", "qkv_ddd")
    labels = ("BBB", "Q", "K", "V", "QK", "QKV")
    x = np.arange(len(labels))
    for rep, color, offset in zip(("0", "1"), COLORS, (-width / 2, width / 2)):
        axes[0].bar(x + offset, [_item_percent(report, rep, name) for name in qkv],
                    width, color=color, label=f"seed {rep}")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Only donor V transfers the key", loc="left", fontweight="bold")
    axes[0].set_ylabel("recipient-specific donor answers (%)")
    axes[0].legend(frameon=False)

    sources = ("queried_f", "other_f", "non_f", "random", "mismatch")
    labels = ("Queried F", "Other F", "Non-F", "Random", "Mismatch")
    x = np.arange(len(labels))
    for rep, color, offset in zip(("0", "1"), COLORS, (-width / 2, width / 2)):
        axes[1].bar(x + offset, [_item_percent(report, rep, name) for name in sources],
                    width, color=color)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_title("The matching F row is the source", loc="left", fontweight="bold")
    axes[1].set_ylabel("recipient-specific donor answers (%)")

    labels, values = [], []
    for rep in ("0", "1"):
        for head in ("1", "3"):
            mass = report["metrics"][rep]["attention_mass"]["base"][head]
            labels.append(f"s{rep}·h{head}")
            values.append((100 * mass["queried_f"], 100 * mass["other_f"],
                           100 * (mass["g_rows"] + mass["marker"] + mass["query"])))
    x = np.arange(len(labels))
    queried = np.array([row[0] for row in values])
    other = np.array([row[1] for row in values])
    rest = np.array([row[2] for row in values])
    axes[2].bar(x, queried, color="#6554C0", label="queried F")
    axes[2].bar(x, other, bottom=queried, color="#AAB4C8", label="other F")
    axes[2].bar(x, rest, bottom=queried + other, color="#E5E8EF", label="remaining")
    axes[2].set_xticks(x, labels)
    axes[2].set_title("Heads already address the right row", loc="left", fontweight="bold")
    axes[2].set_ylabel("base attention mass (%)")
    axes[2].legend(frameon=False, loc="lower right")

    for axis in axes:
        axis.set_ylim(0, 105)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9DEE8", linewidth=.8, alpha=.75)
        axis.set_axisbelow(True)
    fig.suptitle("TEACH-0009 · Stable address, value-stream key transport",
                 fontsize=16, fontweight="bold")
    fig.savefig(RESULT_DIR / "qkv-source-summary.png", dpi=180, facecolor="white")
    fig.savefig(RESULT_DIR / "qkv-source-summary.svg", facecolor="white")


if __name__ == "__main__":
    main()
