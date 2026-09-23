#!/usr/bin/env python3
"""Render a descriptive TEACH-0011 summary from the audited report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "TEACH-0011"
COLORS = ("#2457C5", "#E05A47")


def _pct(report, rep, name):
    return 100 * report["metrics"][rep]["scores"][name]["donor_items"]["accuracy"]


def main():
    report = json.loads((RESULT_DIR / "report.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    width = .36

    qkv = ("qkv_bbb", "qkv_dbb", "qkv_bdb", "qkv_bbd", "qkv_ddb", "qkv_ddd")
    labels = ("BBB", "Q", "K", "V", "QK", "QKV")
    x = np.arange(len(labels))
    for rep, color, offset in zip(("0", "1"), COLORS, (-width / 2, width / 2)):
        axes[0].bar(x + offset, [_pct(report, rep, name) for name in qkv],
                    width, color=color, label=f"seed {rep}")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Changing Q alone switches the object", loc="left", fontweight="bold")
    axes[0].set_ylabel("recipient-specific donor answers (%)")
    axes[0].legend(frameon=False)

    sources = ("both_g", "donor_key_g", "base_key_g", "non_g", "random",
               "cyclic_recipient")
    labels = ("Both G", "Donor G", "Base G", "Non-G", "Random", "Wrong G")
    x = np.arange(len(labels))
    for rep, color, offset in zip(("0", "1"), COLORS, (-width / 2, width / 2)):
        axes[1].bar(x + offset, [_pct(report, rep, name) for name in sources],
                    width, color=color)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_title("Both G-row changes form the edge", loc="left", fontweight="bold")
    axes[1].set_ylabel("recipient-specific donor answers (%)")

    labels, base_mass, edited_mass = [], [], []
    for rep in ("0", "1"):
        for head in ("0", "1", "2", "3"):
            labels.append(f"s{rep}·h{head}")
            base_mass.append(100 * report["metrics"][rep]["attention_mass"]["base"][head][
                "base_key_g"])
            edited_mass.append(100 * report["metrics"][rep]["attention_mass"]["edited"][head][
                "donor_key_g"])
    x = np.arange(len(labels))
    axes[2].plot(x, base_mass, "o-", color="#6554C0", linewidth=2, label="base→base G")
    axes[2].plot(x, edited_mass, "o-", color="#00A88F", linewidth=2,
                 label="edited→donor G")
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].set_title("Every head redirects to the new key", loc="left", fontweight="bold")
    axes[2].set_ylabel("attention mass on matching G row (%)")
    axes[2].legend(frameon=False, loc="lower left")

    for axis in axes:
        axis.set_ylim(0, 105)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9DEE8", linewidth=.8, alpha=.75)
        axis.set_axisbelow(True)
    fig.suptitle("TEACH-0011 · Query-key routing selects the object row",
                 fontsize=16, fontweight="bold")
    fig.savefig(RESULT_DIR / "g-edge-summary.png", dpi=180, facecolor="white")
    svg_path = RESULT_DIR / "g-edge-summary.svg"
    fig.savefig(svg_path, facecolor="white")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
