
import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from src.dsprites.data import StereoDSprites
from src.dsprites.models import ImageEncoder, ImageDecoder, ProjectionHead, LinearProbe
from src.dsprites.losses import off_diagonal, vicreg_loss, deepcca_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_per_sample_isotropic_score(x, num_slices=1024, num_integration_points=17):
    """Epps-Pulley characteristic function test for isotropy"""
    B, K = x.shape
    device = x.device
    
    # 1. Generate random 1D directions (A) and normalize them
    A = torch.randn(K, num_slices, device=device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8)
    
    # 2. Project samples onto these directions: shape (B, num_slices)
    projections = torch.matmul(x, A)
    
    # 3. Setup integration for Epps-Pulley characteristic function
    t = torch.linspace(-5, 5, num_integration_points, device=device)
    target_cf = torch.exp(-0.5 * t**2)
    
    # 4. Calculate the empirical characteristic function (ECF) per sample
    proj_expanded = projections.unsqueeze(-1) # (B, M, 1)
    t_expanded = t.view(1, 1, -1)             # (1, 1, T)
    
    args = proj_expanded * t_expanded
    ecf_real = torch.cos(args)
    ecf_imag = torch.sin(args)
    
    # 5. Calculate squared distance to target CF at each point t
    target_cf_expanded = target_cf.view(1, 1, -1)
    dist_sq = (ecf_real - target_cf_expanded)**2 + ecf_imag**2
    
    # 6. Apply the Gaussian window weighting
    weighted_dist = dist_sq * target_cf_expanded
    
    # 7. Integrate over t and average over slices
    sample_scores = weighted_dist.mean(dim=-1).mean(dim=-1)
    return sample_scores

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = StereoDSprites(root=args.data_root, download=True, 
                             strong_noise_std=args.strong_noise_std, 
        weak_noise_std=args.weak_noise_std,
        jitter_sigma=args.jitter_sigma,
        pos_range=args.pos_range
    )
    val_dataset = StereoDSprites(
        root=args.data_root, 
        n_samples=min(args.n_samples // 10, 10000), 
        strong_noise_std=args.strong_noise_std, 
        weak_noise_std=args.weak_noise_std,
        jitter_sigma=args.jitter_sigma,
        pos_range=args.pos_range
    )# Use a smaller subset for faster experimentation if needed, or full dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Validation loader for evaluation
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Models
    enc_x = ImageEncoder(output_dim=args.latent_dim).to(device)
    enc_y = ImageEncoder(args.latent_dim).to(device)

    decoder_cp = None
    if args.mode in ['cp', 'joint']:
        decoder_cp = ImageDecoder(input_dim=args.latent_dim).to(device)

    proj_x, proj_y = None, None
    if args.mode in ['ca', 'joint', 'deepcca']:
        proj_x = ProjectionHead(args.latent_dim, args.proj_dim).to(device)
        proj_y = ProjectionHead(args.latent_dim, args.proj_dim).to(device)

    classifier = None
    if args.mode == 'supervised':
        classifier = LinearProbe(args.latent_dim, 3).to(device)

    # Optimizer
    if args.mode == 'supervised':
        params = list(enc_x.parameters()) + list(classifier.parameters())
    else:
        params = list(enc_x.parameters()) + list(enc_y.parameters())
        if decoder_cp: params += list(decoder_cp.parameters())
        if proj_x: params += list(proj_x.parameters()) + list(proj_y.parameters())

    optimizer = optim.Adam(params, lr=args.lr)
    criterion_mse = nn.MSELoss()

    print(f"Starting training in {args.mode} mode...")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for epoch in range(args.epochs):
        enc_x.train()
        enc_y.train()
        if classifier:
            classifier.train()
        total_loss = 0

        for batch_idx, (x, y, latents) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            if args.mode == 'supervised':
                lbl = latents[:, 0].long().to(device)
                z_x = enc_x(x)
                loss = F.cross_entropy(classifier(z_x), lbl)
            else:
                z_x = enc_x(x)
                z_y = enc_y(y)

                loss = 0

                # --- Cross-Alignment ---
                if args.mode in ['ca', 'joint']:
                    p_x = proj_x(z_x)
                    p_y = proj_y(z_y)
                    loss += vicreg_loss(p_x, p_y) * (args.lambda_val if args.mode == 'joint' else 1.0)
                elif args.mode == 'deepcca':
                    p_x = proj_x(z_x)
                    p_y = proj_y(z_y)
                    loss += deepcca_loss(p_x, p_y, outdim_size=args.proj_dim)

                # --- Cross-Prediction ---
                if args.mode in ['cp', 'joint']:
                    if args.cp_target == 'y':
                        y_pred = decoder_cp(z_x)
                        cp_loss = criterion_mse(y_pred, y)
                    else:
                        x_pred = decoder_cp(z_y)
                        cp_loss = criterion_mse(x_pred, x)
                    loss += cp_loss

            loss.backward()
            if args.mode == 'deepcca':
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            
            # if batch_idx % 50 == 0:
            #     print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} Average Loss: {total_loss / len(train_loader):.4f}")

        # --- Validation & Early Stopping ---
        enc_x.eval()
        enc_y.eval()
        if args.mode in ['cp', 'joint']:
            decoder_cp.eval()
        if classifier:
            classifier.eval()

        val_loss = 0
        with torch.no_grad():
            for x, y, latents in val_loader:
                x, y = x.to(device), y.to(device)
                if args.mode == 'supervised':
                    lbl = latents[:, 0].long().to(device)
                    loss = F.cross_entropy(classifier(enc_x(x)), lbl)
                else:
                    z_x = enc_x(x)
                    z_y = enc_y(y)
                    loss = 0
                    if args.mode in ['ca', 'joint']:
                        p_x = proj_x(z_x)
                        p_y = proj_y(z_y)
                        loss += vicreg_loss(p_x, p_y) * (args.lambda_val if args.mode == 'joint' else 1.0)
                    elif args.mode == 'deepcca':
                        p_x = proj_x(z_x)
                        p_y = proj_y(z_y)
                        loss += deepcca_loss(p_x, p_y, outdim_size=args.proj_dim)
                    if args.mode in ['cp', 'joint']:
                        if args.cp_target == 'y':
                            y_pred = decoder_cp(z_x)
                            loss += criterion_mse(y_pred, y)
                        else:
                            x_pred = decoder_cp(z_y)
                            loss += criterion_mse(x_pred, x)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            state_dict = {
                'enc_x': enc_x.state_dict(),
                'enc_y': enc_y.state_dict(),
            }
            if args.mode in ['ca', 'joint', 'deepcca']:
                state_dict['proj_x'] = proj_x.state_dict()
                state_dict['proj_y'] = proj_y.state_dict()
            if args.mode in ['cp', 'joint']:
                state_dict['decoder_cp'] = decoder_cp.state_dict()
            if args.mode == 'supervised':
                state_dict['classifier'] = classifier.state_dict()

            torch.save(state_dict, os.path.join(args.save_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # --- Evaluation: Linear Probe on Shape & Isotropy ---
    print("Evaluating Model (Isotropy & Probe)...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    enc_x.load_state_dict(checkpoint['enc_x'])
    enc_y.load_state_dict(checkpoint['enc_y'])
    if args.mode in ['ca', 'joint', 'deepcca']:
        proj_x.load_state_dict(checkpoint['proj_x'])
        proj_y.load_state_dict(checkpoint['proj_y'])
    if args.mode in ['cp', 'joint']:
        decoder_cp.load_state_dict(checkpoint['decoder_cp'])
    if args.mode == 'supervised':
        classifier.load_state_dict(checkpoint['classifier'])

    for p in enc_x.parameters(): p.requires_grad = False
    for p in enc_y.parameters(): p.requires_grad = False
    enc_x.eval()
    enc_y.eval()
    if args.mode in ['ca', 'joint', 'deepcca']:
        proj_x.eval()
        proj_y.eval()
    print("Extracting features for Evaluation...")
    def get_features(loader):
        features = []
        labels = []
        joint_features = []
        
        with torch.no_grad():
            for x, y, latents in loader:
                x = x.to(device)
                y = y.to(device)
                
                if args.mode == 'cp' and args.cp_target == 'x':
                    # We trained on Y -> X, so EncY is the primary encoder
                    z = enc_y(y)
                else:
                    # Default is EncX
                    z = enc_x(x)
                
                # For isotropy, we want the "joint representation" space
                if args.mode in ['ca', 'joint', 'deepcca']:
                    p = proj_x(z)
                else:
                    p = z # For CP, projection is identity
                
                shape_labels = latents[:, 0].numpy()
                features.append(z.cpu().numpy())
                joint_features.append(p.cpu().numpy())
                labels.append(shape_labels)
                
        return np.concatenate(features), np.concatenate(joint_features), np.concatenate(labels)

    z_train, p_train, y_train = get_features(train_loader)
    z_val, p_val, y_val = get_features(val_loader)
    
    # Calculate Isotropy on val set joint embeddings
    p_val_tensor = torch.from_numpy(p_val).to(device)
    p_centered = p_val_tensor - p_val_tensor.mean(dim=0)
    global_std = torch.std(p_centered) + 1e-8
    p_std = p_centered / global_std
    
    isotropy_scores = get_per_sample_isotropic_score(p_std, num_slices=1024)
    mean_isotropy = isotropy_scores.mean().item()
    print(f"Mean Isotropy Score: {mean_isotropy:.4f}")

    # --- UMAP ---
    try:
        import umap
        print("Generating UMAP...")
        reducer = umap.UMAP()
        emb = reducer.fit_transform(z_val)
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(emb[:, 0], emb[:, 1], c=y_val, cmap='viridis', s=5)
        plt.legend(handles=scatter.legend_elements()[0], labels=['Square', 'Ellipse', 'Heart'])
        mode_str = args.mode if args.mode != 'cp' or args.cp_target == 'y' else 'cp_wrong'
        plt.title(f"UMAP - {mode_str} | Samp {args.n_samples} | Jit {args.jitter_sigma} | Noise {args.weak_noise_std} | PosR {args.pos_range}")
        plt.savefig(os.path.join(args.save_dir, 'umap_shape.png'))
        plt.close()
    except ImportError:
        print("umap-learn not installed, skipping UMAP generation.")

    # --- Probe Sweeps ---
    print("Training Linear Probes on different sample sizes...")
    # Log scale from 100 to 10k
    probe_sizes = np.logspace(2, 4, num=10, dtype=int)
    # Ensure sizes don't exceed train set size
    probe_sizes = [s for s in probe_sizes if s <= len(z_train)]

    for p_size in probe_sizes:
        # Sample subset
        idx = np.random.choice(len(z_train), p_size, replace=False)
        z_sub = torch.from_numpy(z_train[idx]).to(device)
        y_sub = torch.from_numpy(y_train[idx]).long().to(device)
        
        # Train probe (more epochs for tiny sizes, but fast since cached)
        probe = LinearProbe(args.latent_dim, 3).to(device)
        probe_opt = optim.Adam(probe.parameters(), lr=1e-3)
        probe_crit = nn.CrossEntropyLoss()
        
        for ep in range(50):
            logits = probe(z_sub)
            loss = probe_crit(logits, y_sub)
            probe_opt.zero_grad()
            loss.backward()
            probe_opt.step()
            
        # Eval on val set
        with torch.no_grad():
            z_val_t = torch.from_numpy(z_val).to(device)
            y_val_t = torch.from_numpy(y_val).long().to(device)
            logits_val = probe(z_val_t)
            preds = logits_val.argmax(dim=1)
            correct = (preds == y_val_t).sum().item()
            acc = correct / len(y_val)
            
        print(f"Probe Size: {p_size:5d} -> Acc: {acc:.4f}")
        
        # Append to master CSV
        if args.results_csv:
            row = pd.DataFrame([{
                'mode': args.mode if args.mode != 'cp' or args.cp_target == 'y' else 'cp_wrong',
                'n_samples': args.n_samples,
                'jitter_sigma': args.jitter_sigma,
                'weak_noise_std': args.weak_noise_std,
                'pos_range': args.pos_range,
                'best_val_loss': best_val_loss,
                'isotropy_score': mean_isotropy,
                'probe_size': p_size,
                'accuracy': acc
            }])
            if not os.path.isfile(args.results_csv):
                row.to_csv(args.results_csv, index=False)
            else:
                row.to_csv(args.results_csv, mode='a', header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./src/dsprites')
    parser.add_argument('--mode', type=str, choices=['ca', 'cp', 'joint', 'deepcca', 'supervised'], default='ca')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=32)
    parser.add_argument('--jitter_sigma', type=float, default=0.05)
    parser.add_argument('--strong_noise_std', type=float, default=0.1)
    parser.add_argument('--weak_noise_std', type=float, default=0.2)
    parser.add_argument('--pos_range', type=float, default=1.0)
    parser.add_argument('--cp_target', type=str, default='y', choices=['x', 'y'], help="Modality to predict in CP mode")
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--lambda_val', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--results_csv', type=str, default='checkpoints/all_results.csv')
    
    args = parser.parse_args()
    train(args)
