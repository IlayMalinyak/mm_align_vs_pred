"""
Combined Aggregated Performance Comparison: dSprites + Shapes3D
===============================================================
Side-by-side panels showing Linear Probe Accuracy vs Nuisance Alignment
for CA and CP methods at n_samples=100k, with shared legend.

Usage:
    python -m src.plot_combined_synthetic [--dsprites_csv ...] [--shapes3d_csv ...]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dsprites_csv", default="checkpoints/all_results.csv")
    p.add_argument("--shapes3d_csv", default="checkpoints/shapes3d_results.csv")
    p.add_argument("--n_samples", type=int, default=100000)
    p.add_argument("--save_dir", default="src/logs/figures/")
    return p.parse_args()


# ── Visual style ─────────────────────────────────────────────────────────
NOISE_COLORS = {
    0.2: "#0072B2",   # blue  (colorblind-safe, Okabe-Ito)
    0.5: "#D55E00",   # vermillion
    0.9: "#009E73",   # bluish green
}

METHOD_STYLE = {
    "CA": {"linestyle": "-",  "marker": "o", "linewidth": 2.5},
    "CP": {"linestyle": "--", "marker": "s", "linewidth": 2.5},
}

NOISE_LABELS = {0.2: "Low", 0.5: "Medium", 0.9: "High"}


def prepare_data(csv_path, n_samples):
    """Load CSV, filter to CA/CP at given n_samples, compute nu_max and max accuracy per run."""
    df = pd.read_csv(csv_path)
    df["pos_range"] = df["pos_range"].fillna(1.0)

    df = df[(df["pos_range"] == 1.0) & (df["mode"].isin(["ca", "cp"]))].copy()
    if n_samples is not None:
        df = df[df["n_samples"] == n_samples]

    df["Method"] = df["mode"].map({"ca": "CA", "cp": "CP"})

    # Max accuracy per run (group by everything except probe_size)
    df_agg = (
        df.groupby(["Method", "n_samples", "jitter_sigma", "weak_noise_std", "best_val_loss"])
        .agg(max_accuracy=("accuracy", "max"))
        .reset_index()
    )
    df_agg["nu_max"] = 1.0 - df_agg["jitter_sigma"]
    return df_agg


def prepare_supervised(csv_path, n_samples):
    """Load supervised ceiling: max accuracy per run, averaged across noise levels per jitter."""
    df = pd.read_csv(csv_path)
    df["pos_range"] = df["pos_range"].fillna(1.0)
    df = df[(df["pos_range"] == 1.0) & (df["mode"] == "supervised")].copy()
    if n_samples is not None:
        df = df[df["n_samples"] == n_samples]
    if df.empty:
        return None

    # Max accuracy across probe sizes per run
    df_agg = (
        df.groupby(["n_samples", "jitter_sigma", "weak_noise_std", "best_val_loss"])
        .agg(max_accuracy=("accuracy", "max"))
        .reset_index()
    )
    df_agg["nu_max"] = 1.0 - df_agg["jitter_sigma"]
    # Average across noise levels (supervised ceiling is noise-invariant)
    stats = df_agg.groupby("nu_max")["max_accuracy"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0)
    return stats


def plot_panel(ax, df, title, sup_stats=None):
    """Draw one panel: accuracy vs nu_max, colored by noise, styled by method."""
    noise_levels = sorted(df["weak_noise_std"].unique())

    # Supervised ceiling (drawn first, behind data lines)
    if sup_stats is not None:
        ax.plot(
            sup_stats["nu_max"], sup_stats["mean"],
            color="#555555", linestyle=":", linewidth=2.0,
            marker="^", markersize=7, zorder=3,
        )
        ax.fill_between(
            sup_stats["nu_max"],
            sup_stats["mean"] - sup_stats["std"],
            sup_stats["mean"] + sup_stats["std"],
            color="#555555", alpha=0.10,
        )

    for method, mstyle in METHOD_STYLE.items():
        for noise in noise_levels:
            sub = df[(df["Method"] == method) & (df["weak_noise_std"] == noise)]
            if sub.empty:
                continue

            # Aggregate across seeds: mean ± std grouped by nu_max
            stats = sub.groupby("nu_max")["max_accuracy"].agg(["mean", "std"]).reset_index()
            stats["std"] = stats["std"].fillna(0)

            ax.plot(
                stats["nu_max"], stats["mean"],
                color=NOISE_COLORS[noise],
                marker=mstyle["marker"],
                linestyle=mstyle["linestyle"],
                linewidth=mstyle["linewidth"],
                markersize=9,
            )
            ax.fill_between(
                stats["nu_max"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                color=NOISE_COLORS[noise],
                alpha=0.12,
            )

    ax.set_xlabel(r"Nuisance Alignment  ($1 - \sigma_{\mathrm{jitter}}$)", fontsize=13)
    ax.set_ylim(0.25, 1.02)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xticks(sorted(df["nu_max"].unique()))


def main():
    args = parse_args()

    # ── Publication style ────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.5,
        "lines.markersize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    # ── Load data ────────────────────────────────────────────────────────
    df_dsp = prepare_data(args.dsprites_csv, args.n_samples)
    df_s3d = prepare_data(args.shapes3d_csv, args.n_samples)

    if df_dsp.empty or df_s3d.empty:
        print(f"Error: no data for n_samples={args.n_samples}.")
        if df_dsp.empty:
            print("  dSprites CSV is empty after filtering.")
        if df_s3d.empty:
            print("  Shapes3D CSV is empty after filtering.")
        return

    # ── Load supervised ceiling ───────────────────────────────────────────
    sup_dsp = prepare_supervised(args.dsprites_csv, args.n_samples)
    sup_s3d = prepare_supervised(args.shapes3d_csv, args.n_samples)

    # ── Create figure ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    plot_panel(ax1, df_dsp, "dSprites", sup_stats=sup_dsp)
    plot_panel(ax2, df_s3d, "Shapes3D", sup_stats=sup_s3d)

    ax1.set_ylabel("Linear Probe Accuracy", fontsize=13)

    # ── Shared legend: two groups (Method + Noise) ─────────────────────
    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([], [], color="black", linestyle="-", marker="o", markersize=8, linewidth=2.5, label="CA"),
        Line2D([], [], color="black", linestyle="--", marker="s", markersize=8, linewidth=2.5, label="CP"),
        Line2D([], [], color="#555555", linestyle=":", marker="^", markersize=7, linewidth=2.0, label="Supervised"),
    ]
    noise_handles = [
        Line2D([], [], color=NOISE_COLORS[n], linestyle="-", linewidth=3, label=NOISE_LABELS[n])
        for n in sorted(NOISE_COLORS)
    ]

    all_handles = method_handles + noise_handles
    all_labels = [h.get_label() for h in all_handles]

    fig.tight_layout()

    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(all_handles),
        frameon=True,
        edgecolor="0.8",
        fontsize=12,
        columnspacing=2.0,
        handlelength=2.5,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"combined_synthetic_comparison_{args.n_samples}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
