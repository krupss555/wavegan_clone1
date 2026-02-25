"""
utils.py – Visualisation and checkpoint helpers.

STFT spectrogram
  window 32, hop 8  (suitable for 128-sample signals)
  Saves one real vs. fake comparison PNG per epoch.

Checkpointing
  Saves G, D, Conditioner state-dicts + optimiser states + metadata.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed on remote SSH
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# STFT visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _stft_power(signal_np, win=32, hop=8):
    """
    Compute STFT magnitude² on a 1-D real-valued signal.
    Returns 2-D array (freq_bins, time_frames).
    Uses numpy hand-rolled STFT so there is no extra dependency.
    """
    n = len(signal_np)
    window = np.hanning(win)
    frames = []
    for start in range(0, n - win + 1, hop):
        frame = signal_np[start: start + win] * window
        spectrum = np.fft.rfft(frame)
        frames.append(np.abs(spectrum) ** 2)
    return np.array(frames).T   # (freq_bins, time_frames)


def save_spectrogram_comparison(real_iq, fake_iq, epoch: int, out_dir: str):
    """
    Plot STFT spectrograms of one real and one fake IQ sample (I channel).

    Parameters
    ----------
    real_iq : torch.Tensor or np.ndarray  shape (2, 128)
    fake_iq : torch.Tensor or np.ndarray  shape (2, 128)
    epoch   : current epoch number (used in filename)
    out_dir : directory where the PNG is saved
    """
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(real_iq, torch.Tensor):
        real_iq = real_iq.detach().cpu().numpy()
    if isinstance(fake_iq, torch.Tensor):
        fake_iq = fake_iq.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(f"Epoch {epoch} – STFT Spectrograms (I and Q channels)")

    labels = ["Real", "Fake"]
    signals = [real_iq, fake_iq]
    ch_names = ["I channel", "Q channel"]

    for col, (sig, lbl) in enumerate(zip(signals, labels)):
        for row, ch in enumerate([0, 1]):
            spec = _stft_power(sig[ch], win=32, hop=8)
            axes[row][col].imshow(
                10 * np.log10(spec + 1e-8),
                aspect="auto", origin="lower",
                cmap="viridis"
            )
            axes[row][col].set_title(f"{lbl} – {ch_names[ch]}")
            axes[row][col].set_xlabel("Time frame")
            axes[row][col].set_ylabel("Freq bin")

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

    # ── Print signal statistics alongside the saved PNG ───────────────────
    def _signal_stats(sig_np, name):
        """sig_np: (2, 128)"""
        amp = np.abs(sig_np[0] + 1j * sig_np[1])
        mag2 = _stft_power(sig_np[0])
        snr  = 10 * np.log10(mag2.max() / (np.percentile(mag2, 10) + 1e-12))
        print(f"  [{name}]  mean_amp={amp.mean():.4f}  "
              f"max_amp={amp.max():.4f}  "
              f"SNR≈{snr:.1f}dB")

    if isinstance(real_iq, torch.Tensor):
        r = real_iq.detach().cpu().numpy()
    else:
        r = real_iq
    if isinstance(fake_iq, torch.Tensor):
        f = fake_iq.detach().cpu().numpy()
    else:
        f = fake_iq

    print(f"[Vis] Epoch {epoch:04d} → {save_path}")
    _signal_stats(r, "Real")
    _signal_stats(f, "Fake")


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(out_dir, epoch, step, G, D, conditioner, G_opt, D_opt):
    """Save a full training checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt")
    torch.save({
        "epoch":           epoch,
        "step":            step,
        "G_state":         G.state_dict(),
        "D_state":         D.state_dict(),
        "cond_state":      conditioner.state_dict(),
        "G_opt_state":     G_opt.state_dict(),
        "D_opt_state":     D_opt.state_dict(),
    }, path)
    print(f"[Ckpt] Saved → {path}")
    return path


def load_checkpoint(path, G, D, conditioner, G_opt, D_opt, device="cpu"):
    """Restore a checkpoint.  Returns (epoch, step)."""
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    conditioner.load_state_dict(ckpt["cond_state"])
    G_opt.load_state_dict(ckpt["G_opt_state"])
    D_opt.load_state_dict(ckpt["D_opt_state"])
    print(f"[Ckpt] Loaded epoch {ckpt['epoch']}, step {ckpt['step']} from {path}")
    return ckpt["epoch"], ckpt["step"]