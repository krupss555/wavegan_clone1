"""
eval/inception_score.py
───────────────────────
Compute Inception Score (IS) and Fréchet Inception Distance (FID)
for generated radar signals, using the RadarClassifier as the feature extractor.

Must run train_classifier.py first to get classifier.pt.

Metrics
───────
IS  – measures diversity + quality of generated signals.
      IS = exp( E_x[ KL(p(y|x) || p(y)) ] )
      Higher is better.

FID – Fréchet distance between real and fake feature distributions.
      FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2·sqrtm(Σ_r·Σ_f))
      Lower is better.

Run
───
python eval/inception_score.py \
    --data_dir      /path/to/data/DeepRadar \
    --ckpt          ./runs/phase1/checkpoints/ckpt_epoch_0100.pt \
    --classifier    ./runs/eval/classifier.pt \
    --n_samples     5000 \
    --batch         128 \
    --latent        100 \
    --emb_dim       32 \
    --splits        10
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model   import Generator, PhaseOneConditioner
from dataset import DeepRadarDataset
from eval.train_classifier import RadarClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features_and_probs(model, loader_or_tensors, device):
    """
    Returns:
      feats  (N, 256)  intermediate feature vectors
      probs  (N, 23)   softmax class probabilities
    """
    model.eval()
    all_feats = []
    all_probs = []

    for batch in loader_or_tensors:
        if isinstance(batch, (list, tuple)):
            x = batch[0]   # dataset returns (signal, label)
        else:
            x = batch
        x = x.to(device)
        feats  = model.features(x)                        # (B, 256)
        logits = model.fc(feats)                          # (B, 23)
        probs  = F.softmax(logits, dim=-1)                # (B, 23)
        all_feats.append(feats.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_feats, axis=0), np.concatenate(all_probs, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Inception Score
# ──────────────────────────────────────────────────────────────────────────────

def inception_score(probs, splits=10):
    """
    probs : (N, C) softmax probabilities
    Returns: mean IS, std IS over `splits` splits
    """
    N = probs.shape[0]
    split_size = N // splits
    scores = []
    for k in range(splits):
        p = probs[k * split_size: (k + 1) * split_size]   # (M, C)
        p_y = p.mean(axis=0, keepdims=True)                # marginal (1, C)
        kl  = p * (np.log(p + 1e-8) - np.log(p_y + 1e-8))
        kl  = kl.sum(axis=1).mean()                        # scalar
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


# ──────────────────────────────────────────────────────────────────────────────
# FID
# ──────────────────────────────────────────────────────────────────────────────

def _sqrtm(A):
    """Matrix square-root via eigendecomposition (CPU numpy, symmetric PSD)."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)   # numerical stability
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def frechet_distance(feats_real, feats_fake):
    """
    feats_real, feats_fake : (N, D) float32/64 numpy arrays
    Returns scalar FID.
    """
    mu_r, mu_f = feats_real.mean(0), feats_fake.mean(0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_f = np.cov(feats_fake, rowvar=False)

    diff      = mu_r - mu_f
    covmean   = _sqrtm(sigma_r @ sigma_f)

    # handle numerical imaginary parts
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


# ──────────────────────────────────────────────────────────────────────────────
# Generate fake samples tensor (batched)
# ──────────────────────────────────────────────────────────────────────────────

def generate_fake_loader(G, conditioner, n_samples, latent_dim, batch_size, device):
    """
    Returns a list of (batch_tensor,) tuples that mimic a DataLoader.
    Labels are sampled uniformly so all 23 classes are represented.
    """
    G.eval(); conditioner.eval()
    batches = []
    generated = 0
    with torch.no_grad():
        while generated < n_samples:
            bsz    = min(batch_size, n_samples - generated)
            z      = torch.randn(bsz, latent_dim, device=device)
            labels = torch.randint(0, 23, (bsz,), device=device)
            emb    = conditioner(labels)
            fake   = G(z, emb).cpu()   # (B, 2, 128)
            batches.append((fake,))
            generated += bsz
    return batches


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[IS/FID] device={device}")

    # ── Load classifier ───────────────────────────────────────────────────
    print(f"[IS/FID] Loading classifier from {args.classifier} …")
    clf_ckpt = torch.load(args.classifier, map_location=device)
    clf      = RadarClassifier(num_classes=clf_ckpt["num_classes"]).to(device)
    clf.load_state_dict(clf_ckpt["model_state"])
    clf.eval()
    print(f"  Classifier val_acc = {clf_ckpt['val_acc']*100:.2f}%")

    # ── Load Generator ────────────────────────────────────────────────────
    print(f"[IS/FID] Loading generator from {args.ckpt} …")
    ckpt = torch.load(args.ckpt, map_location=device)
    G    = Generator(latent_dim=args.latent, emb_dim=args.emb_dim).to(device)
    G.load_state_dict(ckpt["G_state"])
    cond = PhaseOneConditioner(num_classes=23, emb_dim=args.emb_dim).to(device)
    cond.load_state_dict(ckpt["cond_state"])
    print(f"  Loaded checkpoint  epoch={ckpt['epoch']}  step={ckpt['step']}")

    # ── Real features ─────────────────────────────────────────────────────
    print(f"[IS/FID] Extracting REAL features …")
    real_ds     = DeepRadarDataset(args.data_dir, split="val")
    real_loader = DataLoader(real_ds, batch_size=args.batch,
                             shuffle=False, num_workers=4)
    feats_real, _ = extract_features_and_probs(clf, real_loader, device)
    print(f"  Real features : {feats_real.shape}")

    # ── Fake samples + features ───────────────────────────────────────────
    print(f"[IS/FID] Generating {args.n_samples} FAKE samples …")
    fake_loader = generate_fake_loader(
        G, cond, args.n_samples, args.latent, args.batch, device
    )
    feats_fake, probs_fake = extract_features_and_probs(clf, fake_loader, device)
    print(f"  Fake features : {feats_fake.shape}")

    # ── Inception Score ───────────────────────────────────────────────────
    is_mean, is_std = inception_score(probs_fake, splits=args.splits)
    print(f"\n{'='*50}")
    print(f"  Inception Score : {is_mean:.4f}  ±  {is_std:.4f}")

    # ── FID ───────────────────────────────────────────────────────────────
    # align sample counts (use min)
    n = min(len(feats_real), len(feats_fake))
    fid = frechet_distance(feats_real[:n].astype(np.float64),
                           feats_fake[:n].astype(np.float64))
    print(f"  FID             : {fid:.4f}")
    print(f"{'='*50}\n")
    print("Interpretation:")
    print("  IS  → higher is better  (max ≈ 23 = num classes)")
    print("  FID → lower  is better  (0 = identical distributions)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--ckpt",       required=True,  help="Generator checkpoint .pt")
    p.add_argument("--classifier", required=True,  help="RadarClassifier checkpoint .pt")
    p.add_argument("--n_samples",  type=int,   default=5000)
    p.add_argument("--batch",      type=int,   default=128)
    p.add_argument("--latent",     type=int,   default=100)
    p.add_argument("--emb_dim",    type=int,   default=32)
    p.add_argument("--splits",     type=int,   default=10,
                   help="Number of splits for IS estimation")
    args = p.parse_args()
    evaluate(args)