"""
eval/noise.py
─────────────
Evaluate noise characteristics of generated radar signals.

Metrics
───────
1. Noise floor (dB)
2. Spectral flatness (Wiener entropy)
3. SNR estimate (dB)
4. Amplitude distribution similarity (KS statistic)
5. Phase distribution similarity (Earth Mover Distance)

Saves:
  <out_dir>/noise_eval_epoch<EPOCH>.png
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Generator, PhaseOneConditioner
from dataset import DeepRadarDataset
from eval.inception_score import generate_fake_loader


# ─────────────────────────────────────────────────────────────
# STFT
# ─────────────────────────────────────────────────────────────

def _stft_mag(signal_1d, win=32, hop=8):
    window = np.hanning(win)
    frames = []
    for s in range(0, len(signal_1d) - win + 1, hop):
        frame = signal_1d[s:s+win] * window
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    return np.array(frames).T


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def noise_floor_db(mag2):
    flat = mag2.ravel()
    thresh = np.percentile(flat, 10)
    noise = flat[flat <= thresh]
    return 10 * np.log10(noise.mean() + 1e-12)


def spectral_flatness(mag2):
    power = mag2.mean(axis=1) + 1e-12
    gm = np.exp(np.log(power).mean())
    am = power.mean()
    return float(gm / am)


def snr_db(mag2):
    flat = mag2.ravel()
    peak = flat.max()
    noise = np.percentile(flat, 10)
    return 10 * np.log10((peak + 1e-12) / (noise + 1e-12))


def compute_signal_metrics(signals_np):
    noise_floors, flatness_vals, snr_vals = [], [], []
    amps, phases, stft_all = [], [], []

    for sig in signals_np:
        mag2 = _stft_mag(sig[0])
        stft_all.append(mag2.ravel())

        noise_floors.append(noise_floor_db(mag2))
        flatness_vals.append(spectral_flatness(mag2))
        snr_vals.append(snr_db(mag2))

        complex_sig = sig[0] + 1j * sig[1]
        amps.append(np.abs(complex_sig))
        phases.append(np.angle(complex_sig))

    return {
        "noise_floor_db": float(np.mean(noise_floors)),
        "spectral_flatness": float(np.mean(flatness_vals)),
        "snr_db": float(np.mean(snr_vals)),
        "amplitudes": np.concatenate(amps),
        "phases": np.concatenate(phases),
        "stft_power": np.concatenate(stft_all),
    }


# ─────────────────────────────────────────────────────────────
# Distribution Metrics
# ─────────────────────────────────────────────────────────────

def ks_statistic(a, b):
    a_sort = np.sort(a)
    b_sort = np.sort(b)
    combined = np.concatenate([a_sort, b_sort])
    cdf_a = np.searchsorted(a_sort, combined, side="right") / len(a)
    cdf_b = np.searchsorted(b_sort, combined, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def phase_emd(real_phase, fake_phase, bins=100):
    hist_r, bin_edges = np.histogram(real_phase, bins=bins, range=(-np.pi, np.pi), density=True)
    hist_f, _ = np.histogram(fake_phase, bins=bins, range=(-np.pi, np.pi), density=True)

    cdf_r = np.cumsum(hist_r)
    cdf_f = np.cumsum(hist_f)

    emd = np.mean(np.abs(cdf_r - cdf_f))
    return float(emd)


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def save_noise_figure(real_m, fake_m, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Epoch {epoch} – Noise Evaluation", fontsize=13)

    # Amplitude CDF
    ax = axes[0]
    for label, amps in [("Real", real_m["amplitudes"]),
                        ("Fake", fake_m["amplitudes"])]:
        s = np.sort(amps)
        ax.plot(s, np.linspace(0, 1, len(s)), label=label)
    ax.set_title("Amplitude CDF")
    ax.set_xlabel("|I + jQ|")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # STFT histogram (ALL samples)
    ax = axes[1]
    combined = np.concatenate([real_m["stft_power"], fake_m["stft_power"]])
    bins = np.linspace(0, np.percentile(combined, 95), 60)

    ax.hist(real_m["stft_power"], bins=bins, density=True, alpha=0.5, label="Real")
    ax.hist(fake_m["stft_power"], bins=bins, density=True, alpha=0.5, label="Fake")

    ax.set_title("STFT Power Distribution")
    ax.set_xlabel("Power")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"noise_eval_epoch{epoch:04d}.png")
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"[Noise] Saved figure → {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Noise] device={device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    G = Generator(latent_dim=args.latent, emb_dim=args.emb_dim).to(device)
    G.load_state_dict(ckpt["G_state"])
    cond = PhaseOneConditioner(num_classes=23, emb_dim=args.emb_dim).to(device)
    cond.load_state_dict(ckpt["cond_state"])
    epoch_loaded = ckpt["epoch"]
    print(f"[Noise] Loaded checkpoint epoch={epoch_loaded}")

    # Real signals
    real_ds = DeepRadarDataset(args.data_dir, split="val")
    real_loader = DataLoader(real_ds, batch_size=args.batch, shuffle=False)
    real_list = []
    for x, _ in real_loader:
        real_list.append(x.numpy())
        if sum(len(r) for r in real_list) >= args.n_samples:
            break
    real_signals = np.concatenate(real_list)[:args.n_samples]

    # Fake signals
    fake_batches = generate_fake_loader(
        G, cond, args.n_samples, args.latent, args.batch, device
    )
    fake_list = [b[0].numpy() for b in fake_batches]
    fake_signals = np.concatenate(fake_list)[:args.n_samples]

    # Compute metrics
    real_m = compute_signal_metrics(real_signals)
    fake_m = compute_signal_metrics(fake_signals)

    ks = ks_statistic(real_m["amplitudes"], fake_m["amplitudes"])
    emd = phase_emd(real_m["phases"], fake_m["phases"])

    print(f"\n{'='*60}")
    print(f"{'Metric':<25} {'Real':>10}  {'Fake':>10}  {'Diff':>10}")
    print(f"{'-'*60}")
    for key in ["noise_floor_db", "spectral_flatness", "snr_db"]:
        r, f = real_m[key], fake_m[key]
        unit = " dB" if "db" in key else ""
        print(f"{key:<25} {r:>9.3f}{unit}  {f:>9.3f}{unit}  {f-r:>+9.3f}{unit}")
    print(f"{'Amplitude KS stat':<25} {'–':>10}  {'–':>10}  {ks:>10.4f}")
    print(f"{'Phase EMD':<25} {'–':>10}  {'–':>10}  {emd:>10.4f}")
    print(f"{'='*60}\n")

    save_noise_figure(real_m, fake_m, epoch_loaded, args.out_dir)


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="./runs/eval")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--latent", type=int, default=100)
    p.add_argument("--emb_dim", type=int, default=32)
    args = p.parse_args()
    evaluate(args)