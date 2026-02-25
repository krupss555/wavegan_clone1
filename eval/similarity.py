"""
eval/similarity.py
──────────────────
Measure the perceptual similarity between REAL and FAKE radar signals
using k-nearest-neighbour distances in the classifier feature space.
Replaces TF eval/similarity/feats.py + eval/similarity/sim.py.

Two metrics are computed:
─────────────────────────
1. NDB  (Number of statistically Different Bins)
   Cluster real features with k-means.
   Assign each fake feature to the nearest cluster.
   Bins whose real/fake proportions differ significantly (2-sample z-test)
   are counted.  Lower NDB → more coverage.

2. KNN Precision / Recall
   Precision: fraction of fake samples that fall inside the real manifold
              (near at least one real sample in feature space)
   Recall   : fraction of real samples that have at least one nearby fake
   Both in [0,1].  Higher is better.

Run
───
python eval/similarity.py \
    --data_dir   /path/to/data/DeepRadar \
    --ckpt       ./runs/phase1/checkpoints/ckpt_epoch_0100.pt \
    --classifier ./runs/eval/classifier.pt \
    --n_samples  5000 \
    --batch      128 \
    --latent     100 \
    --emb_dim    32 \
    --k          5 \
    --n_clusters 10 \
    --ndb_alpha  0.05
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model   import Generator, PhaseOneConditioner
from dataset import DeepRadarDataset
from eval.train_classifier import RadarClassifier
from eval.inception_score  import (extract_features_and_probs,
                                   generate_fake_loader)


# ──────────────────────────────────────────────────────────────────────────────
# KNN helpers (pure numpy, no sklearn needed)
# ──────────────────────────────────────────────────────────────────────────────

def _pairwise_l2_sq(A, B, chunk=500):
    """
    Compute squared L2 distance matrix between A (N,D) and B (M,D).
    Processes A in chunks to keep memory manageable.
    Returns (N, M) float32 array.
    """
    N, M = len(A), len(B)
    out  = np.empty((N, M), dtype=np.float32)
    BB   = (B ** 2).sum(1)          # (M,)
    for start in range(0, N, chunk):
        end  = min(start + chunk, N)
        a    = A[start:end]                                # (c, D)
        aa   = (a ** 2).sum(1, keepdims=True)              # (c, 1)
        out[start:end] = aa + BB - 2 * (a @ B.T)          # (c, M)
    return np.maximum(out, 0)


def knn_precision_recall(feats_real, feats_fake, k=5):
    """
    Improved Precision and Recall (Kynkäänniemi et al. 2019).
    Precision: fraction of fake in real k-NN balls.
    Recall   : fraction of real in fake k-NN balls.
    """
    # k-NN thresholds on real manifold
    D_rr = _pairwise_l2_sq(feats_real, feats_real)
    np.fill_diagonal(D_rr, np.inf)
    knn_real_thresh = np.partition(D_rr, k, axis=1)[:, k - 1]   # (N_r,)

    # k-NN thresholds on fake manifold
    D_ff = _pairwise_l2_sq(feats_fake, feats_fake)
    np.fill_diagonal(D_ff, np.inf)
    knn_fake_thresh = np.partition(D_ff, k, axis=1)[:, k - 1]   # (N_f,)

    # Precision: for each fake, is it within any real ball?
    D_fr = _pairwise_l2_sq(feats_fake, feats_real)               # (N_f, N_r)
    in_real_ball = (D_fr <= knn_real_thresh[None, :]).any(axis=1)
    precision    = in_real_ball.mean()

    # Recall: for each real, is it within any fake ball?
    D_rf = D_fr.T                                                 # (N_r, N_f)
    in_fake_ball = (D_rf <= knn_fake_thresh[None, :]).any(axis=1)
    recall       = in_fake_ball.mean()

    return float(precision), float(recall)


# ──────────────────────────────────────────────────────────────────────────────
# k-means (pure numpy)
# ──────────────────────────────────────────────────────────────────────────────

def kmeans(X, k, max_iter=100, seed=42):
    """Lloyd's k-means.  Returns (labels, centroids)."""
    rng       = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    labels    = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        D       = _pairwise_l2_sq(X, centroids)   # (N, k)
        new_lbl = D.argmin(axis=1)
        if (new_lbl == labels).all():
            break
        labels = new_lbl
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = X[mask].mean(0)
    return labels, centroids


# ──────────────────────────────────────────────────────────────────────────────
# NDB
# ──────────────────────────────────────────────────────────────────────────────

def ndb_score(feats_real, feats_fake, n_clusters=10, alpha=0.05):
    """
    Number of statistically Different Bins.
    Returns (ndb, js_divergence).
    """
    # cluster real features
    real_labels, centroids = kmeans(feats_real, k=n_clusters)

    # assign fake to nearest centroid
    D_fc     = _pairwise_l2_sq(feats_fake, centroids)
    fake_labels = D_fc.argmin(axis=1)

    N_r, N_f = len(feats_real), len(feats_fake)
    real_counts = np.bincount(real_labels, minlength=n_clusters) / N_r
    fake_counts = np.bincount(fake_labels, minlength=n_clusters) / N_f

    # 2-sample binomial z-test per bin
    ndb = 0
    for b in range(n_clusters):
        p_r = real_counts[b]
        p_f = fake_counts[b]
        p_pool = (p_r * N_r + p_f * N_f) / (N_r + N_f)
        se     = np.sqrt(p_pool * (1 - p_pool) * (1/N_r + 1/N_f)) + 1e-9
        z      = abs(p_r - p_f) / se
        # z > 1.96 ≈ α=0.05 two-tailed
        if z > 1.96:
            ndb += 1

    # JS divergence as bonus metric
    m = 0.5 * (real_counts + fake_counts) + 1e-10
    js = 0.5 * (
        (real_counts * np.log((real_counts + 1e-10) / m)).sum() +
        (fake_counts * np.log((fake_counts + 1e-10) / m)).sum()
    )

    return ndb, float(js)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Similarity] device={device}")

    # ── Load classifier ───────────────────────────────────────────────────
    print(f"[Similarity] Loading classifier from {args.classifier} …")
    clf_ckpt = torch.load(args.classifier, map_location=device)
    clf      = RadarClassifier(clf_ckpt["num_classes"]).to(device)
    clf.load_state_dict(clf_ckpt["model_state"])
    clf.eval()

    # ── Load Generator ────────────────────────────────────────────────────
    print(f"[Similarity] Loading generator from {args.ckpt} …")
    ckpt = torch.load(args.ckpt, map_location=device)
    G    = Generator(latent_dim=args.latent, emb_dim=args.emb_dim).to(device)
    G.load_state_dict(ckpt["G_state"])
    cond = PhaseOneConditioner(num_classes=23, emb_dim=args.emb_dim).to(device)
    cond.load_state_dict(ckpt["cond_state"])

    # ── Real features ─────────────────────────────────────────────────────
    print("[Similarity] Extracting REAL features …")
    real_ds     = DeepRadarDataset(args.data_dir, split="val")
    real_loader = DataLoader(real_ds, batch_size=args.batch,
                             shuffle=False, num_workers=4)
    feats_real, _ = extract_features_and_probs(clf, real_loader, device)
    print(f"  Real features: {feats_real.shape}")

    # ── Fake features ─────────────────────────────────────────────────────
    print(f"[Similarity] Generating {args.n_samples} FAKE samples …")
    fake_loader = generate_fake_loader(
        G, cond, args.n_samples, args.latent, args.batch, device
    )
    feats_fake, _ = extract_features_and_probs(clf, fake_loader, device)
    print(f"  Fake features: {feats_fake.shape}")

    # ── KNN Precision / Recall ────────────────────────────────────────────
    print(f"\n[Similarity] Computing k={args.k} NN precision/recall …")
    n = min(len(feats_real), len(feats_fake))
    precision, recall = knn_precision_recall(
        feats_real[:n], feats_fake[:n], k=args.k
    )

    # ── NDB ───────────────────────────────────────────────────────────────
    print(f"[Similarity] Computing NDB with {args.n_clusters} clusters …")
    ndb, js = ndb_score(feats_real[:n], feats_fake[:n],
                        n_clusters=args.n_clusters, alpha=args.ndb_alpha)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  KNN Precision     : {precision:.4f}   (↑ higher = fakes inside real manifold)")
    print(f"  KNN Recall        : {recall:.4f}   (↑ higher = real manifold covered by fakes)")
    print(f"  NDB               : {ndb}/{args.n_clusters}     (↓ lower = distributions match)")
    print(f"  JS Divergence     : {js:.4f}   (↓ lower = more similar)")
    print(f"{'='*50}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--classifier", required=True)
    p.add_argument("--n_samples",  type=int,   default=5000)
    p.add_argument("--batch",      type=int,   default=128)
    p.add_argument("--latent",     type=int,   default=100)
    p.add_argument("--emb_dim",    type=int,   default=32)
    p.add_argument("--k",          type=int,   default=5,
                   help="k for kNN precision/recall")
    p.add_argument("--n_clusters", type=int,   default=10,
                   help="k-means clusters for NDB")
    p.add_argument("--ndb_alpha",  type=float, default=0.05)
    args = p.parse_args()
    evaluate(args)