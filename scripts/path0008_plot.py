"""Descriptive confirmation figure for the frozen PATH-0008 result."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("results/PATH-0008")


def run() -> None:
    summary = json.loads((ROOT / "summary.json").read_text())
    decision = json.loads((ROOT / "decision.json").read_text())
    assert decision["status"] == "exploratory_failed"
    n = summary["upstream_success"]
    assert n == 32
    conditions = [
        ("One selected head", "selected_one", "#a4b7c4"),
        ("Four selected heads", "selected_four", "#c46e42"),
        ("Random edit, same size", "random_four_norm", "#b6b5ad"),
        ("Other bundle, same size", "mismatch_four_norm", "#b6b5ad"),
        ("All heads in selected block", "all_heads", "#6689a0"),
        ("Full four-block window", "full_window", "#227b70"),
        ("All later attention (control)", "all_cut", "#8b9b9f"),
    ]
    counts = [summary["reductions"][key] for _, key, _ in conditions]
    fig, ax = plt.subplots(figsize=(10.2, 5.5), dpi=180)
    y = list(range(len(conditions)))
    ax.barh(y, [count / n for count in counts], color=[color for _, _, color in conditions],
            height=0.62)
    for index, count in enumerate(counts):
        ax.text(min(count / n + .016, 1.015), index, f"{count}/{n}", va="center",
                fontsize=10.5, color="#233a46", fontweight="semibold")
    ax.axvline(.35, color="#a6563d", linestyle="--", linewidth=1.3)
    ax.text(.35, -.72, "registered minimum for selected four", ha="left", va="center",
            color="#a6563d", fontsize=9)
    ax.set_yticks(y, [label for label, _, _ in conditions], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fresh cases where edited answer's first token disappeared", fontsize=10.5)
    ax.grid(axis="x", color="#e3e8ea", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("The causal effect spans more than the selected four heads", x=.06,
                 y=.98, ha="left", fontsize=15, fontweight="bold", color="#173143")
    fig.text(.06, .025,
             "32 fresh direct-lookup cases in 8 dependent lexical bundles. Supplied English tables; no Voynich reading.",
             fontsize=9, color="#546b73")
    fig.subplots_adjust(left=.30, right=.95, top=.85, bottom=.16)
    fig.savefig(ROOT / "head-vs-window-confirmation.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    run()
