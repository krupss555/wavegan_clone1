"""
train_phase1.py – Phase 1 WaveGAN pre-training on DeepRadar2022.

WGAN-GP training:
  • 5 discriminator updates per generator update
  • gradient penalty λ = 10
  • Adam  lr=1e-4, β=(0.5, 0.9)
  • batch = 64

Logging   : every 100 iterations  (epoch | batch | G_loss | D_loss)
Checkpoint: every epoch  →  <out_dir>/checkpoints/
Vis       : every epoch  →  <out_dir>/vis/  (STFT PNG)

Run (single command, no intermediate steps):
───────────────────────────────────────────────────────────────────────────────
python train_phase1.py \
    --data_dir  /path/to/data/DeepRadar \
    --out_dir   ./runs/phase1 \
    --epochs    200 \
    --batch     64 \
    --latent    100 \
    --emb_dim   32 \
    --n_critic  5 \
    --lr        1e-4 \
    --gp_lambda 10 \
    --num_workers 4 \
    --resume    ""
───────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# local imports
from model   import Generator, Discriminator, PhaseOneConditioner
from dataset import DeepRadarDataset
from utils   import save_checkpoint, load_checkpoint, save_spectrogram_comparison


# ──────────────────────────────────────────────────────────────────────────────
# WGAN-GP gradient penalty
# ──────────────────────────────────────────────────────────────────────────────

def gradient_penalty(D, real, fake, cond_emb_detached, device, lam=10.0):
    """
    Compute WGAN-GP gradient penalty.
    cond_emb is detached so GP only penalises wrt the interpolated signal.
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp = D(interp, cond_emb_detached)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # (B, C, T)

    grads = grads.view(B, -1)                          # (B, C*T)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return lam * gp


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ── Directories ──────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    vis_dir  = os.path.join(args.out_dir, "vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)

    # ── Dataset & Loader ─────────────────────────────────────────────────────
    train_ds = DeepRadarDataset(args.data_dir, split="train", crop_len=128)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,        # keeps batch size constant (needed for GP)
    )

    val_ds = DeepRadarDataset(args.data_dir, split="val", crop_len=128)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ── Models ───────────────────────────────────────────────────────────────
    G         = Generator(latent_dim=args.latent, emb_dim=args.emb_dim).to(device)
    D         = Discriminator(emb_dim=args.emb_dim, phaseshuffle_rad=2).to(device)
    conditioner = PhaseOneConditioner(num_classes=23, emb_dim=args.emb_dim).to(device)

    # ── Optimisers ───────────────────────────────────────────────────────────
    G_opt = torch.optim.Adam(
        list(G.parameters()) + list(conditioner.parameters()),
        lr=args.lr, betas=(0.5, 0.9)
    )
    D_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step  = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, G, D, conditioner, G_opt, D_opt, device=str(device)
        )
        start_epoch += 1   # resume from next epoch

    # ── Training ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train(); conditioner.train()

        d_loss_acc = 0.0
        g_loss_acc = 0.0
        batch_count = 0

        # keep one real sample for visualisation at end of epoch
        vis_real = None

        for batch_idx, (real_sig, labels) in enumerate(train_loader):
            real_sig = real_sig.to(device)       # (B, 2, 128)
            labels   = labels.to(device)         # (B,)

            # ── D update (n_critic times) ────────────────────────────────
            for _ in range(args.n_critic):
                D_opt.zero_grad()

                # condition embedding (detach so D doesn't tune conditioner)
                cond_emb = conditioner(labels).detach()

                # fake signal
                z = torch.randn(args.batch, args.latent, device=device)
                with torch.no_grad():
                    fake_sig = G(z, cond_emb)

                d_real  = D(real_sig, cond_emb)
                d_fake  = D(fake_sig.detach(), cond_emb)
                gp      = gradient_penalty(D, real_sig, fake_sig.detach(),
                                           cond_emb, device, lam=args.gp_lambda)

                d_loss = d_fake.mean() - d_real.mean() + gp
                d_loss.backward()
                D_opt.step()

            d_loss_acc += d_loss.item()

            # ── G update ─────────────────────────────────────────────────
            G_opt.zero_grad()

            cond_emb_g = conditioner(labels)
            z          = torch.randn(args.batch, args.latent, device=device)
            fake_sig_g = G(z, cond_emb_g)

            # freeze D during G backward (no .step() called)
            g_loss = -D(fake_sig_g, cond_emb_g.detach()).mean()
            g_loss.backward()
            G_opt.step()

            g_loss_acc += g_loss.item()
            global_step += 1
            batch_count += 1

            # ── Logging every 100 iterations ─────────────────────────────
            if global_step % 100 == 0:
                avg_d = d_loss_acc / batch_count
                avg_g = g_loss_acc / batch_count
                print(
                    f"Epoch {epoch:04d} | "
                    f"Batch {batch_idx + 1:05d}/{len(train_loader)} | "
                    f"G_loss {avg_g:+.4f} | "
                    f"D_loss {avg_d:+.4f} | "
                    f"step {global_step}"
                )
                d_loss_acc = 0.0
                g_loss_acc = 0.0
                batch_count = 0

            # cache one real sample for vis
            if vis_real is None:
                vis_real = real_sig[0].detach()

        # ── End-of-epoch visualisation ────────────────────────────────────
        G.eval()
        with torch.no_grad():
            vis_label = torch.zeros(1, dtype=torch.long, device=device)
            vis_cond  = conditioner(vis_label)
            vis_z     = torch.randn(1, args.latent, device=device)
            vis_fake  = G(vis_z, vis_cond)[0].detach()  # (2, 128)

        save_spectrogram_comparison(
            real_iq=vis_real,
            fake_iq=vis_fake,
            epoch=epoch,
            out_dir=vis_dir,
        )

        # ── End-of-epoch checkpoint ───────────────────────────────────────
        save_checkpoint(
            ckpt_dir, epoch, global_step,
            G, D, conditioner, G_opt, D_opt
        )

        # ── Validation D-loss (one pass) ──────────────────────────────────
        G.eval(); D.eval(); conditioner.eval()
        val_d_losses = []
        with torch.no_grad():
            for real_sig_v, labels_v in val_loader:
                real_sig_v = real_sig_v.to(device)
                labels_v   = labels_v.to(device)
                cond_v     = conditioner(labels_v)
                z_v        = torch.randn(args.batch, args.latent, device=device)
                fake_v     = G(z_v, cond_v)
                d_real_v   = D(real_sig_v, cond_v).mean().item()
                d_fake_v   = D(fake_v,     cond_v).mean().item()
                val_d_losses.append(d_fake_v - d_real_v)

        val_d = float(np.mean(val_d_losses))
        print(f"[Val] Epoch {epoch:04d} | D_loss (Wasserstein estimate) {val_d:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="WaveGAN Phase-1 training on DeepRadar2022"
    )
    p.add_argument("--data_dir",     type=str,   required=True,
                   help="Path to DeepRadar directory (contains X_train.mat etc.)")
    p.add_argument("--out_dir",      type=str,   default="./runs/phase1",
                   help="Output directory for checkpoints and visualisations")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch",        type=int,   default=64)
    p.add_argument("--latent",       type=int,   default=100,
                   help="Latent vector dimension (z)")
    p.add_argument("--emb_dim",      type=int,   default=32,
                   help="Condition embedding dimension (must match Phase 2)")
    p.add_argument("--n_critic",     type=int,   default=5,
                   help="Discriminator updates per generator update")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--gp_lambda",    type=float, default=10.0,
                   help="Gradient penalty coefficient")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--resume",       type=str,   default="",
                   help="Path to checkpoint .pt file to resume from (leave empty to start fresh)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("WaveGAN Phase-1 – DeepRadar2022 pre-training")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:20s}: {v}")
    print("=" * 70)
    train(args)