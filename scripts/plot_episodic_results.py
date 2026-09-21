"""Render audited synthetic results; charts never recompute scientific decisions."""

import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "raw_fixed": "Transformer / fixed tasks",
    "raw_fresh": "Transformer / fresh tasks",
    "canonical_fixed": "Canonical / fixed tasks",
    "canonical_fresh": "Canonical / fresh tasks",
    "canonical_large": "Canonical / larger model",
    "gru_fresh": "GRU / fresh tasks",
    "signed_delta_fresh": "Signed recurrence / fresh tasks",
}
FAMILIAR = {"cycle", "branch", "pair_parity", "iid"}
COLORS = ["#9BA9B5", "#2864A8", "#C4B3A4", "#168B81", "#6B52A3", "#C28A2C", "#BC5663"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/episodic-20260921"))
    parser.add_argument("--out", type=Path, default=Path("results/episodic-20260921/EXP-0012"))
    args = parser.parse_args()
    report = json.loads((args.results / "EXP-0012/confirmation_report.json").read_text())
    causal = json.loads((args.results / "EXP-0013/summary.json").read_text())
    runs = report["results"]
    means, seed_means, beyond, durations = [], [], [], []
    for key in LABELS:
        selected = [r for r in runs if r["run"].startswith(key + "-s")]
        values = [np.mean([t["model_bits"] for t in r["tasks"] if t["family"] in FAMILIAR]) for r in selected]
        means.append(np.mean(values))
        seed_means.append(values)
        beyond.append(np.mean([t["beyond_first_kl_bits"] for r in selected for t in r["tasks"]]))
        durations.append(
            sum(
                json.loads((args.results / "EXP-0012" / (r["run"] + "__summary.json")).read_text())[
                    "elapsed_seconds"
                ]
                for r in selected
            )
            / 60
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0, 0]
    y = np.arange(len(LABELS))
    ax.scatter(means, y, c=COLORS, s=95, zorder=3)
    for i, values in enumerate(seed_means):
        ax.plot(values, [i] * len(values), "|", color="#142D40", ms=11, alpha=0.75)
        ax.text(max(values) + 0.008, i, f"{means[i]:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, list(LABELS.values()))
    ax.invert_yaxis()
    ax.set_xlim(min(means) - 0.04, max(means) + 0.08)
    ax.set_xlabel("Bits per symbol on new tasks from familiar families · lower is better")
    ax.set_title("Does the training strategy transfer?", loc="left", pad=16)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.text(
        0,
        -0.18,
        "Dots: mean across tasks and seeds. Ticks: three training seeds.\nSame maximum budget; selected checkpoints may use different updates.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        color="#52606D",
    )
    ax = axes[0, 1]
    ax.barh(y, beyond, color=COLORS, height=0.65)
    ax.set_yticks(y, [""] * len(y))
    ax.invert_yaxis()
    ax.set_xlabel("Joint KL beyond the first symbol · bits · lower is better")
    ax.set_title("Do later dependencies match the oracle?", loc="left", pad=16)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.text(
        0,
        -0.18,
        "All six families; exact three-symbol futures.\nThe oracle knows the process parameters; the models do not.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        color="#52606D",
    )
    ax = axes[1, 0]
    ax.barh(y, durations, color=COLORS, height=0.65)
    ax.set_yticks(y, list(LABELS.values()))
    ax.invert_yaxis()
    ax.set_xlabel("Summed measured run time across three seeds · minutes")
    ax.set_title("Where the compute went", loc="left", pad=16)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.text(
        0,
        -0.18,
        "Completed runs only; interrupted attempts excluded.\nCPU and GPU overlap; these bars are not elapsed campaign time.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        color="#52606D",
    )
    ax = axes[1, 1]
    method_names, values, teachers = [], [], []
    for family in ["gru_fresh", "signed_delta_fresh"]:
        for seed in [121, 122, 123]:
            r = json.loads((args.results / "EXP-0013" / f"{family}-s{seed}-r4.json").read_text())
            section = r["confirmation"]
            try:
                base = section["unchanged"]["immediate_matched_pairs"]["donor_conditional_after_first_bits"][
                    "equal_key_mean"
                ]
                value = section["learned"]["immediate_matched_pairs"]["donor_conditional_after_first_bits"][
                    "equal_key_mean"
                ]
                ratio = value / base if base is not None and base >= 0.01 and value is not None else np.nan
            except (KeyError, TypeError):
                ratio = np.nan
            method_names.append(("GRU" if family == "gru_fresh" else "Signed") + f" / seed {seed}")
            values.append(ratio)
            teachers.append(r["teacher_qualification"])
    yy = np.arange(len(values))
    ax.barh(yy, values, color=["#168B81" if q else "#9BA9B5" for q in teachers])
    for i, value in enumerate(values):
        if not np.isfinite(value):
            ax.text(0.02, i, "insufficient baseline gap", va="center", fontsize=9)
    ax.axvline(1, color="#C24242", linestyle="--", linewidth=1)
    ax.set_yticks(yy, method_names)
    ax.invert_yaxis()
    ax.set_xlabel("Remaining donor discrepancy / unchanged discrepancy")
    ax.set_title("A narrow state change faces stricter tests", loc="left", pad=16)
    statuses = "; ".join(
        ("GRU" if k == "gru_fresh" else "Signed") + ": " + v["status"].replace("_", " ")
        for k, v in causal["primary_families"].items()
    )
    ax.text(
        0,
        -0.18,
        "Primary gate: "
        + statuses
        + "\nGray: teacher does not meet the oracle-fidelity gate.\nRatios require baseline gap >= 0.01 bits; no ratio alone establishes success.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        color="#52606D",
    )
    fig.suptitle(
        "Learning unfamiliar rules: EPISODIC-0012 + 0013",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.06,
        0.947,
        "Synthetic process inference and model interpretation — no manuscript decoding",
        fontsize=12,
        color="#52606D",
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.89, bottom=0.12, hspace=0.6, wspace=0.36)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(args.out / f"overview.{ext}", dpi=180, facecolor="white", bbox_inches="tight")
    svg = args.out / "overview.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    sources = [
        args.results / "EXP-0012/confirmation_report.json",
        args.results / "EXP-0013/summary.json",
        *sorted((args.results / "EXP-0012").glob("*__summary.json")),
        *sorted((args.results / "EXP-0013").glob("*-r4.json")),
    ]
    (args.out / "figure_provenance.json").write_text(
        json.dumps(
            {
                "matplotlib": matplotlib.__version__,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
                },
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "no_gate_recomputation": True,
            },
            indent=2,
        )
        + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
