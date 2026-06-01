import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='checkpoints/all_results.csv')
    parser.add_argument('--save_dir', type=str, default='src/logs/figures/dsprites/')
    return parser.parse_args()

MODE_MAPPING = {
    'ca': 'CA',
    'cp': 'CP (Strong to Weak)',
    'cp_wrong': 'CP (Weak to Strong)',
    'deepcca': 'CA (DeepCCA)'
}

def plot_results(csv_path, save_dir):
    # Set publication-ready style
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
    
    # We want 3 plots, one for each n_samples
    df['pos_range'] = df['pos_range'].fillna(1.0)
    
    modes = ['ca', 'cp', 'cp_wrong', 'deepcca']
    df['mode_display'] = df['mode'].map(MODE_MAPPING)
    
    samples_levels = sorted(df['n_samples'].unique())
    noise_levels = sorted(df['weak_noise_std'].unique())
    
    for n_samp in samples_levels:
        df_samp = df[(df['n_samples'] == n_samp) & (df['pos_range'] == 1.0)]
        
        # Create 3x3 grid (Rows: weak_noise_std, Cols: mode)
        fig, axes = plt.subplots(len(noise_levels), len(modes), figsize=(18, 4 * len(noise_levels)), sharex=True, sharey=True)
        
        for i, noise in enumerate(noise_levels):
            for j, mode in enumerate(modes):
                ax = axes[i, j] if len(noise_levels) > 1 else axes[j]
                # Filter data
                df_sub = df_samp[(df_samp['weak_noise_std'] == noise) & (df_samp['mode'] == mode)]
                
                if df_sub.empty:
                    continue
                
                # Plot acc vs probe_size for each jitter_sigma
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
                
                # Titles and labels
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
                    
                # Clean up legend
                if i == 0 and j == 1:
                    ax.legend(title=r'Jitter $\sigma$', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                        
        plt.suptitle(f"Shape Classification Accuracy (Total Pretrain Samples: {n_samp})", fontweight='bold')
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
    
    # Group by all experiment params EXCEPT seed to get observations for lineplot
    # We first get the max accuracy per individual run (seed)
    df_run_max = df.groupby(['mode', 'mode_display', 'n_samples', 'jitter_sigma', 'weak_noise_std', 'pos_range', 'best_val_loss']).agg(
        max_accuracy=('accuracy', 'max')
    ).reset_index()

    noise_levels = sorted(df_run_max['weak_noise_std'].unique())
    
    # 1. Plot Max Accuracy vs n_samples
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(5 * len(noise_levels), 5), sharey=True)
    if len(noise_levels) == 1: axes = [axes]
    
    for i, noise in enumerate(noise_levels):
        ax = axes[i]
        df_sub = df_run_max[df_run_max['weak_noise_std'] == noise]
        
        sns.lineplot(
            data=df_sub, x='n_samples', y='max_accuracy', 
            hue='jitter_sigma', style='mode_display', markers=True, ax=ax, palette='tab10',
            errorbar='sd' # Show standard deviation
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
                
    plt.suptitle("Classification Accuracy vs Total Pretrain Samples", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_samples.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot Best Val Loss vs n_samples
    modes = sorted(df_run_max['mode'].unique())
    fig, axes = plt.subplots(len(modes), len(noise_levels), figsize=(6 * len(noise_levels), 5 * len(modes)))
    if len(modes) == 1: axes = [axes]
    
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
            # Use log scale for loss since CP/CA losses have different scales
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
    
    # For isotropy, we also want to allow lineplot to aggregate over seeds
    # Group by run params (each seed has one isotropy score)
    df_iso_runs = df.groupby(['mode', 'mode_display', 'n_samples', 'jitter_sigma', 'weak_noise_std', 'pos_range']).agg(
        isotropy=('isotropy_score', 'first') # Isotropy is same for all probe_sizes in a run
    ).reset_index()

    noise_levels = sorted(df_iso_runs['weak_noise_std'].unique())
    
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(6 * len(noise_levels), 5), sharey=True)
    if len(noise_levels) == 1: axes = [axes]
    
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
    
    # Filter for N=50k and max probe size
    max_probe = df['probe_size'].max()
    df_sub = df[(df['n_samples'] == 50000) & (df['probe_size'] == max_probe)]
    
    if df_sub['pos_range'].nunique() < 2:
        return

    for jitter_val in [0.0, 0.2, 0.5]:
        df_jit = df_sub[df_sub['jitter_sigma'] == jitter_val]
        if df_jit.empty:
            continue
            
        noises = sorted(df_jit['weak_noise_std'].unique())
        if not noises: continue
        
        fig, axes = plt.subplots(1, len(noises), figsize=(5 * len(noises), 5), sharey=True)
        if len(noises) == 1: axes = [axes]
        
        for ax, noise_val in zip(axes, noises):
            df_plot = df_jit[df_jit['weak_noise_std'] == noise_val]
            if df_plot.empty: continue
            
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
            # Adjust legend so it doesn't overlap excessively 
            ax.legend(title='Mode')
            
        fig.suptitle(f"Sample Efficiency vs. Position Variance | Jitter $\\sigma={jitter_val}$", fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"accuracy_vs_pos_variance_jit{jitter_val}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved position variance plot for jitter {jitter_val} to {save_dir}")

def plot_aggregated_main_experiment(csv_path, save_dir):
    """Plot Accuracy vs Jitter, colored by Noise, styled by Mode, saved as separate figures per n_samples."""
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    df['pos_range'] = df['pos_range'].fillna(1.0)
    
    # Filter for the main sweep (pos_range = 1.0) and modes CA/CP
    df_main = df[(df['pos_range'] == 1.0) & (df['mode'].isin(['ca', 'cp', 'deepcca']))].copy()

    # Rename for this plot
    LOCAL_MODE_MAPPING = {
        'ca': 'CA (VICReg)',
        'cp': 'CP',
        'deepcca': 'CA (DeepCCA)'
    }
    df_main['Method'] = df_main['mode'].map(LOCAL_MODE_MAPPING)
    df_main['Weak Noise Std'] = df_main['weak_noise_std']
    
    # We want max accuracy per run configuration and seed
    df_run_max = df_main.groupby(['Method', 'n_samples', 'jitter_sigma', 'Weak Noise Std', 'best_val_loss']).agg(
        max_accuracy=('accuracy', 'max')
    ).reset_index()

    # Create nu_max column for the x-axis
    df_run_max['$\\nu_{max}$'] = 1.0 - df_run_max['jitter_sigma']

    samples_levels = sorted(df_run_max['n_samples'].unique())

    # Create a smoother palette for the noise lines
    # We have 3 noise levels: 0.2, 0.5, 0.9. A sequential palette is good here to show intensity
    colors = sns.color_palette("mako", n_colors=3)

    for n_samp in samples_levels:
        df_plot = df_run_max[df_run_max['n_samples'] == n_samp]
        if df_plot.empty: continue
        
        plt.figure(figsize=(8, 6))
        
        sns.lineplot(
            data=df_plot,
            x='$\\nu_{max}$',
            y='max_accuracy',
            hue='Weak Noise Std',
            style='Method', # This uses different markers and line dashed for CA vs CP
            palette=colors, # Softer, smoother sequential colormap ("mako")
            markers=['o', 's', 'D'], # Explicit distinct markers for 3 methods
            dashes=True, # Distinguish modes using dashes
            linewidth=2.5,
            markersize=10,
            errorbar='sd'
        )
        
        plt.title(f"Pretrain Samples: {n_samp // 1000}k", fontweight='bold', fontsize=16, pad=15)
        plt.xlabel(r"Nuisance Alignment ($1 - \sigma_{jitter}$)")
        plt.ylabel("Linear Probe Accuracy")
        plt.ylim(0.0, 1.05)
        plt.grid(True, ls='--', alpha=0.5)
        
        # Clean legend
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, title_fontsize=14)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"aggregated_performance_comparison_{n_samp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved aggregated performance comparison ({n_samp} samples) to {save_path}")

def plot_vicreg_vs_deepcca(csv_path, save_dir):
    """Plot VICReg vs DeepCCA comparison (CA modes only, no CP).

    Matches the shapes3d comparison pattern: one figure per n_samples,
    hue=noise, style=Method (VICReg vs DeepCCA).
    """
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df['pos_range'] = df['pos_range'].fillna(1.0)

    # CA modes only — direct VICReg vs DeepCCA comparison
    df_main = df[(df['pos_range'] == 1.0) & (df['mode'].isin(['ca', 'deepcca']))].copy()
    if df_main.empty:
        print("No VICReg+DeepCCA data found — skipping vicreg_vs_deepcca plot")
        return

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
        save_path = os.path.join(save_dir, f"vicreg_vs_deepcca_{n_samp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved VICReg vs DeepCCA comparison ({n_samp} samples) to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    plot_results(args.csv, args.save_dir)
    plot_sample_efficiency(args.csv, args.save_dir)
    plot_isotropy(args.csv, args.save_dir)
    plot_pos_efficiency(args.csv, args.save_dir)
    plot_aggregated_main_experiment(args.csv, args.save_dir)
    plot_vicreg_vs_deepcca(args.csv, args.save_dir)
