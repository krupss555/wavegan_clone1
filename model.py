"""
model.py – PyTorch WaveGAN adapted for 128-sample IQ radar signals.

Architecture (faithful to Donahue et al., scaled to 128 samples):

Generator
  z (100) + cond_emb (32)  →  FC  →  reshape (4, 256)
  → ConvTranspose1d s=4  →  (16, 128)
  → ConvTranspose1d s=4  →  (64,  64)
  → ConvTranspose1d s=2  →  (128,  2)   ← IQ output, tanh

Discriminator
  Input (128, 2+1)  ← cond projected to (1, 128) then cat
  → Conv1d s=4  →  (32,  64)   + phase-shuffle
  → Conv1d s=4  →  (8,  128)   + phase-shuffle
  → Conv1d s=4  →  (2,  256)
  → flatten (512)  →  FC  →  scalar

Conditioning (Phase 1)
  nn.Embedding(24, 32)  for 23 radar modulation classes (0-indexed)
  In discriminator: linear 32→128 and reshape to (1, 128) extra channel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def phase_shuffle(x, rad=2):
    """
    Randomly shift each sample in the batch by ±rad along the time axis
    (reflect-padded so length stays constant).  Applied in discriminator only.
    x: (B, C, T)
    """
    if rad == 0:
        return x
    B, C, T = x.shape
    # one random shift per sample in batch
    shifts = torch.randint(-rad, rad + 1, (B,), device=x.device)
    out = torch.zeros_like(x)
    for i in range(B):
        s = shifts[i].item()
        if s == 0:
            out[i] = x[i]
        elif s > 0:
            # shift right → pad left with reflection
            out[i, :, s:] = x[i, :, :T - s]
            out[i, :, :s] = x[i, :, 1:s + 1].flip(dims=[-1])
        else:  # s < 0
            s = abs(s)
            out[i, :, :T - s] = x[i, :, s:]
            out[i, :, T - s:] = x[i, :, T - s - 1:T - 1].flip(dims=[-1])
    return out


def lrelu(x, alpha=0.2):
    return F.leaky_relu(x, negative_slope=alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Conditional WaveGAN Generator.
    Input:  z (B, latent_dim=100)  +  cond_emb (B, emb_dim=32)
    Output: (B, 2, 128)  IQ waveform in [-1, 1]

    ConvTranspose1d sizes:
      kernel=25, padding=12, output_padding=3 for stride 4
      kernel=25, padding=12, output_padding=1 for stride 2
    """

    def __init__(self, latent_dim=100, emb_dim=32):
        super().__init__()
        inp_dim = latent_dim + emb_dim     # 132

        # FC: 132 → 4*256 = 1024, then reshape to (B, 256, 4)
        self.fc = nn.Linear(inp_dim, 4 * 256)

        # Upsample: (B, 256, 4) → (B, 128, 16) → (B, 64, 64) → (B, 2, 128)
        self.upconv0 = nn.ConvTranspose1d(256, 128, kernel_size=25,
                                          stride=4, padding=12, output_padding=3)
        self.upconv1 = nn.ConvTranspose1d(128,  64, kernel_size=25,
                                          stride=4, padding=12, output_padding=3)
        self.upconv2 = nn.ConvTranspose1d( 64,   2, kernel_size=25,
                                          stride=2, padding=12, output_padding=1)

    def forward(self, z, cond_emb):
        """
        z        : (B, 100)
        cond_emb : (B, 32)
        returns  : (B, 2, 128)
        """
        x = torch.cat([z, cond_emb], dim=1)   # (B, 132)
        x = self.fc(x)                          # (B, 1024)
        x = x.view(x.size(0), 256, 4)          # (B, 256, 4)
        x = F.relu(self.upconv0(x))             # (B, 128, 16)
        x = F.relu(self.upconv1(x))             # (B,  64, 64)
        x = torch.tanh(self.upconv2(x))         # (B,   2, 128)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Discriminator
# ──────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Conditional WaveGAN Discriminator.
    cond_emb (B, 32) is projected to (B, 1, 128) and concatenated as an
    extra channel so the discriminator sees (B, 3, 128).

    Conv1d sizes  (kernel=25, padding=12, stride=4):
      128 → 32 → 8 → 2
    Flatten: 2 * 256 = 512  →  FC → 1
    """

    def __init__(self, emb_dim=32, phaseshuffle_rad=2):
        super().__init__()
        self.rad = phaseshuffle_rad

        # project condition to (1, 128) extra channel
        self.cond_proj = nn.Linear(emb_dim, 128)

        # Input channels: 2 IQ + 1 cond = 3
        self.conv0 = nn.Conv1d(  3,  64, kernel_size=25, stride=4, padding=12)
        self.conv1 = nn.Conv1d( 64, 128, kernel_size=25, stride=4, padding=12)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=25, stride=4, padding=12)

        self.fc = nn.Linear(2 * 256, 1)   # T=2 after 3 downs, 256 ch

    def forward(self, x, cond_emb):
        """
        x        : (B, 2, 128)
        cond_emb : (B, 32)
        returns  : (B,)  raw logit (no sigmoid for WGAN-GP)
        """
        # project condition → (B, 1, 128)
        c = self.cond_proj(cond_emb).unsqueeze(1)   # (B, 1, 128)
        x = torch.cat([x, c], dim=1)                 # (B, 3, 128)

        x = lrelu(self.conv0(x))                      # (B,  64, 32)
        x = phase_shuffle(x, self.rad)
        x = lrelu(self.conv1(x))                      # (B, 128,  8)
        x = phase_shuffle(x, self.rad)
        x = lrelu(self.conv2(x))                      # (B, 256,  2)

        x = x.view(x.size(0), -1)                     # (B, 512)
        return self.fc(x).squeeze(-1)                  # (B,)


# ──────────────────────────────────────────────────────────────────────────────
# Conditioner  –  Phase 1 (discrete classes)
# ──────────────────────────────────────────────────────────────────────────────

class PhaseOneConditioner(nn.Module):
    """
    Maps integer class label (0-22) → embedding vector (B, emb_dim).
    Identical interface to a Phase-2 conditioner so the G/D weights
    never need to change between phases.
    """

    def __init__(self, num_classes=23, emb_dim=32):
        super().__init__()
        # index 0..22 (DeepRadar is 1-indexed → we shift to 0-indexed)
        self.emb = nn.Embedding(num_classes, emb_dim)

    def forward(self, labels):
        """labels: (B,) long tensor, values 0-22"""
        return self.emb(labels)   # (B, 32)