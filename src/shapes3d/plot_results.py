import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='checkpoints/shapes3d_results.csv')
    parser.add_argument('--save_dir', type=str, default='src/logs/figures/shapes3d/')
    return parser.parse_args()


MODE_MAPPING = {
    'ca': 'CA',
    'cp': 'CP (Strong to Weak)',
    'cp_wrong': 'CP (Weak to Strong)',
    'deepcca': 'CA (DeepCCA)'
}


def plot_results(csv_path, save_dir):
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 16,
        "lines.linewidth": 2.5,
        "lines.markersize": 8
    })

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df['pos_range'] = df['pos_range'].fillna(1.0)

    modes = ['ca', 'cp', 'cp_wrong', 'deepcca']
    df['mode_display'] = df['mode'].map(MODE_MAPPING)

    samples_levels = sorted(df['n_samples'].unique())
    noise_levels = sorted(df['weak_noise_std'].unique())

    for n_samp in samples_levels:
        df_samp = df[(df['n_samples'] == n_samp) & (df['pos_range'] == 1.0)]

        fig, axes = plt.subplots(len(noise_levels), len(modes),
                                 figsize=(18, 4 * len(noise_levels)), sharex=True, sharey=True)

        for i, noise in enumerate(noise_levels):
            for j, mode in enumerate(modes):
                ax = axes[i, j] if len(noise_levels) > 1 else axes[j]
                df_sub = df_samp[(df_samp['weak_noise_std'] == noise) & (df_samp['mode'] == mode)]

                if df_sub.empty:
                    continue

                sns.lineplot(
                    data=df_sub,
                    x='probe_size',
                    y='accuracy',
                    hue='jitter_sigma',
                    marker='o',
                    palette='tab10',
                    ax=ax,
                    errorbar='sd'
                )

                ax.set_xscale('log')
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, which='both', ls='--', alpha=0.5)

                if i == 0:
                    ax.set_title(MODE_MAPPING.get(mode, mode.upper()), fontweight='bold')
                if j == 0:
                    ax.set_ylabel(f"Acc (Noise={noise})\n")
                else:
                    ax.set_ylabel("")

                if i == len(noise_levels) - 1:
                    ax.set_xlabel("Linear Probe Train Size")
                else:
                    ax.set_xlabel("")

                if i == 0 and j == 1:
                    ax.legend(title=r'Jitter $\sigma$', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()

        plt.suptitle(f"3D Shape Classification Accuracy (Total Pretrain Samples: {n_samp})", fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"accuracy_grid_samples_{n_samp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")


def plot_sample_efficiency(csv_path, save_dir):
    """Plot Best Val Loss and Max Accuracy vs n_samples"""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df['pos_range'] = df['pos_range'].fillna(1.0)
    df['mode_display'] = df['mode'].map(MODE_MAPPING)

    df_run_max = df.groupby(
        ['mode', 'mode_display', 'n_samples', 'jitter_sigma', 'weak_noise_std', 'pos_range', 'best_val_loss']
    ).agg(max_accuracy=('accuracy', 'max')).reset_index()

    noise_levels = sorted(df_run_max['weak_noise_std'].unique())

    fig, axes = plt.subplots(1, len(noise_levels), figsize=(5 * len(noise_levels), 5), sharey=True)
    if len(noise_levels) == 1:
        axes = [axes]

    for i, noise in enumerate(noise_levels):
        ax = axes[i]
        df_sub = df_run_max[df_run_max['weak_noise_std'] == noise]

        sns.lineplot(
            data=df_sub, x='n_samples', y='max_accuracy',
            hue='jitter_sigma', style='mode_display', markers=True, ax=ax, palette='tab10',
            errorbar='sd'
        )
        ax.set_title(f"Weak Noise Std: {noise}")
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        ax.grid(True, ls='--', alpha=0.5)

        if i == 0:
            ax.set_ylabel("Max Accuracy")
        else:
            ax.set_ylabel("")

        if ax.get_legend() is not None:
            if i == len(noise_levels) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.get_legend().remove()

    plt.suptitle("3D Shape Classification Accuracy vs Total Pretrain Samples", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_samples.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Val loss plot
    modes = sorted(df_run_max['mode'].unique())
    fig, axes = plt.subplots(len(modes), len(noise_levels), figsize=(6 * len(noise_levels), 5 * len(modes)))
    if len(modes) == 1:
        axes = [axes]

    for j, mode in enumerate(modes):
        for i, noise in enumerate(noise_levels):
            ax = axes[j, i] if len(modes) > 1 else axes[i]
            df_sub = df_run_max[(df_run_max['weak_noise_std'] == noise) & (df_run_max['mode'] == mode)]

            sns.lineplot(
                data=df_sub, x='n_samples', y='best_val_loss',
                hue='jitter_sigma', marker='o', ax=ax, palette='tab10',
                errorbar='sd'
            )
            ax.set_title(f"{MODE_MAPPING.get(mode, mode.upper())} | Noise: {noise}", fontweight='bold')
            ax.set_xscale('log')
            if not df_sub.empty and (df_sub['best_val_loss'] > 0).all():
                ax.set_yscale('log')
            ax.grid(True, ls='--', alpha=0.5)

            if ax.get_legend() is not None:
                if i == len(noise_levels) - 1 and j == 0:
                    ax.legend(title=r'Jitter $\sigma$', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax.get_legend().remove()

    plt.suptitle("Validation Loss vs Total Pretrain Samples", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_vs_samples.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sample efficiency plots to {save_dir}")


def plot_isotropy(csv_path, save_dir):
    """Plot Mean Isotropy Score vs n_samples and jitter"""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df['pos_range'] = df['pos_range'].fillna(1.0)
    df['mode_display'] = df['mode'].map(MODE_MAPPING)

    df_iso_runs = df.groupby(
        ['mode', 'mode_display', 'n_samples', 'jitter_sigma', 'weak_noise_std', 'pos_range']
    ).agg(isotropy=('isotropy_score', 'first')).reset_index()

    noise_levels = sorted(df_iso_runs['weak_noise_std'].unique())

    fig, axes = plt.subplots(1, len(noise_levels), figsize=(6 * len(noise_levels), 5), sharey=True)
    if len(noise_levels) == 1:
        axes = [axes]

    for i, noise in enumerate(noise_levels):
        ax = axes[i]
        df_sub = df_iso_runs[df_iso_runs['weak_noise_std'] == noise]

        sns.lineplot(
            data=df_sub, x='n_samples', y='isotropy',
            hue='jitter_sigma', style='mode_display', markers=True, ax=ax, palette='tab10',
            errorbar='sd'
        )
        ax.set_title(f"Weak Noise Std: {noise}", fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, ls='--', alpha=0.5)

        if i == 0:
            ax.set_ylabel("Isotropy Score (Lower is better)")
        else:
            ax.set_ylabel("")

        if ax.get_legend() is not None:
            if i == len(noise_levels) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.get_legend().remove()

    plt.suptitle("Joint Embedding Isotropy vs Total Pretrain Samples", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "isotropy_vs_samples.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved isotropy plot to {save_dir}")


def plot_pos_efficiency(csv_path, save_dir):
    """Plot Accuracy vs Position Range variance"""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df['pos_range'] = df['pos_range'].fillna(1.0)
    df['mode_display'] = df['mode'].map(MODE_MAPPING)

    max_probe = df['probe_size'].max()
    df_sub = df[(df['n_samples'] == 50000) & (df['probe_size'] == max_probe)]

    if df_sub['pos_range'].nunique() < 2:
        return

    for jitter_val in [0.0, 0.2, 0.5]:
        df_jit = df_sub[df_sub['jitter_sigma'] == jitter_val]
        if df_jit.empty:
            continue

        noises = sorted(df_jit['weak_noise_std'].unique())
        if not noises:
            continue

        fig, axes = plt.subplots(1, len(noises), figsize=(5 * len(noises), 5), sharey=True)
        if len(noises) == 1:
            axes = [axes]

        for ax, noise_val in zip(axes, noises):
            df_plot = df_jit[df_jit['weak_noise_std'] == noise_val]
            if df_plot.empty:
                continue

            sns.lineplot(
                data=df_plot, x='pos_range', y='accuracy',
                hue='mode_display', markers=True, palette='tab10',
                errorbar='sd', ax=ax
            )
            ax.set_title(f"Weak Noise $\\sigma={noise_val}$")
            ax.set_xlabel("Position Range (Smaller = Lower Variance)")
            if ax == axes[0]:
                ax.set_ylabel("Linear Probe Accuracy")
            else:
                ax.set_ylabel("")
            ax.grid(True, ls='--', alpha=0.5)
            ax.legend(title='Mode')

        fig.suptitle(f"Sample Efficiency vs. Position Variance | Jitter $\\sigma={jitter_val}$",
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"accuracy_vs_pos_variance_jit{jitter_val}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved position variance plot for jitter {jitter_val} to {save_dir}")


def plot_aggregated_main_experiment(csv_path, save_dir):
    """Plot Accuracy vs Jitter, colored by Noise, styled by Mode."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df['pos_range'] = df['pos_range'].fillna(1.0)

    df_main = df[(df['pos_range'] == 1.0) & (df['mode'].isin(['ca', 'deepcca']))].copy()

    LOCAL_MODE_MAPPING = {'ca': 'CA (VICReg)', 'deepcca': 'CA (DeepCCA)'}
    df_main['Method'] = df_main['mode'].map(LOCAL_MODE_MAPPING)
    df_main['Weak Noise Std'] = df_main['weak_noise_std']

    df_run_max = df_main.groupby(
        ['Method', 'n_samples', 'jitter_sigma', 'Weak Noise Std', 'best_val_loss']
    ).agg(max_accuracy=('accuracy', 'max')).reset_index()

    df_run_max['$\\nu_{max}$'] = 1.0 - df_run_max['jitter_sigma']

    samples_levels = sorted(df_run_max['n_samples'].unique())
    colors = sns.color_palette("mako", n_colors=3)

    for n_samp in samples_levels:
        df_plot = df_run_max[df_run_max['n_samples'] == n_samp]
        if df_plot.empty:
            continue

        plt.figure(figsize=(8, 6))

        sns.lineplot(
            data=df_plot,
            x='$\\nu_{max}$',
            y='max_accuracy',
            hue='Weak Noise Std',
            style='Method',
            palette=colors,
            markers=['o', 's'],
            dashes=True,
            linewidth=2.5,
            markersize=10,
            errorbar='sd'
        )

        plt.title(f"Pretrain Samples: {n_samp // 1000}k", fontweight='bold', fontsize=16, pad=15)
        plt.xlabel(r"Nuisance Alignment ($1 - \sigma_{jitter}$)")
        plt.ylabel("Linear Probe Accuracy")
        plt.ylim(0.0, 1.05)
        plt.grid(True, ls='--', alpha=0.5)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, title_fontsize=14)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"vicreg_vs_deepcca_{n_samp}_shape3d.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved aggregated performance comparison ({n_samp} samples) to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    plot_results(args.csv, args.save_dir)
    plot_sample_efficiency(args.csv, args.save_dir)
    plot_isotropy(args.csv, args.save_dir)
    plot_pos_efficiency(args.csv, args.save_dir)
    plot_aggregated_main_experiment(args.csv, args.save_dir)
