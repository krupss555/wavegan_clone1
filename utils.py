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
    Plot STFT spectrograms and Time-Domain Line Graphs of one real and one fake IQ sample.

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

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Epoch {epoch} – Radar Signal Comparison", fontsize=16)

    time_axis = np.arange(128)

    # ─────────────────────────────────────────────────────────
    # ROW 1: TIME DOMAIN LINE GRAPHS (Real vs Fake Overlay)
    # ─────────────────────────────────────────────────────────
    
    # Top-Left: I Channel
    axes[0, 0].plot(time_axis, real_iq[0], label="Real Signal", color="blue", linewidth=1.5)
    axes[0, 0].plot(time_axis, fake_iq[0], label="Fake Signal", color="red", linewidth=1.5, linestyle="--")
    axes[0, 0].set_title("I-Channel: Time Domain Comparison")
    axes[0, 0].set_xlabel("Time (samples)")
    axes[0, 0].set_ylabel("Normalized Amplitude")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, linestyle=":", alpha=0.7)

    # Top-Right: Q Channel
    axes[0, 1].plot(time_axis, real_iq[1], label="Real Signal", color="darkorange", linewidth=1.5)
    axes[0, 1].plot(time_axis, fake_iq[1], label="Fake Signal", color="green", linewidth=1.5, linestyle="--")
    axes[0, 1].set_title("Q-Channel: Time Domain Comparison")
    axes[0, 1].set_xlabel("Time (samples)")
    axes[0, 1].set_ylabel("Normalized Amplitude")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, linestyle=":", alpha=0.7)

    # ─────────────────────────────────────────────────────────
    # ROW 2: SMOOTHED SPECTROGRAMS (I Channel)
    # ─────────────────────────────────────────────────────────
    
    real_spec = _stft_power(real_iq[0], win=32, hop=8)
    fake_spec = _stft_power(fake_iq[0], win=32, hop=8)

    # Bottom-Left: Real Spectrogram
    im_r = axes[1, 0].imshow(
        10 * np.log10(real_spec + 1e-8),
        aspect="auto", origin="lower", cmap="jet", interpolation="bicubic"
    )
    axes[1, 0].set_title("Real Signal (I-Channel) - Smoothed Spectrogram")
    axes[1, 0].set_xlabel("Time Frame")
    axes[1, 0].set_ylabel("Frequency Bin")
    fig.colorbar(im_r, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Power (dB)")

    # Bottom-Right: Fake Spectrogram
    im_f = axes[1, 1].imshow(
        10 * np.log10(fake_spec + 1e-8),
        aspect="auto", origin="lower", cmap="jet", interpolation="bicubic"
    )
    axes[1, 1].set_title("Fake Signal (I-Channel) - Smoothed Spectrogram")
    axes[1, 1].set_xlabel("Time Frame")
    axes[1, 1].set_ylabel("Frequency Bin")
    fig.colorbar(im_f, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Power (dB)")

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=150)
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

    print(f"[Vis] Epoch {epoch:04d} → Saved visual comparison to {save_path}")
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