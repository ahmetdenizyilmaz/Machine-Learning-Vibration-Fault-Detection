"""
Deep Fusion Network for vibration fault detection.

Early/deep fusion: feature branch (MLP) and signal branch (1D CNN)
exchange information at every layer through gated fusion blocks.

Architecture:
    35 features → FeatureBranch(MLP) → 128-dim ─┐
                                                  ├→ GatedFusion → ResBlock ─┐
    raw signal  → SignalBranch(CNN)  → 128-dim ─┘                            │
                                                  ┌──────────────────────────┘
                                                  ├→ GatedFusion → ResBlock ─┐
                                                  │                          │
                                                  ... (repeated N times)     │
                                                  ┌──────────────────────────┘
                                                  └→ Head → n_classes
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Signal Branch — 1D CNN for raw vibration waveform
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → GELU → MaxPool."""
    def __init__(self, in_ch, out_ch, kernel_size=7, pool=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.net(x)


class SignalBranch(nn.Module):
    """1D CNN that maps a variable-length waveform to a fixed-size vector.

    Input:  (batch, n_axes, signal_length)   — n_axes is 1, 2, or 3
    Output: (batch, out_dim)
    """
    def __init__(self, n_axes=3, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(n_axes, 32,  kernel_size=15, pool=4),   # long kernel for low-freq
            ConvBlock(32,     64,  kernel_size=7,  pool=4),
            ConvBlock(64,     128, kernel_size=5,  pool=2),
            ConvBlock(128,    128, kernel_size=3,  pool=2),
            ConvBlock(128,    128, kernel_size=3,  pool=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, n_axes, L)
        h = self.cnn(x)           # (B, 128, L')
        h = self.pool(h)          # (B, 128, 1)
        h = h.squeeze(-1)         # (B, 128)
        return self.proj(h)       # (B, out_dim)


# ---------------------------------------------------------------------------
# Feature Branch — MLP for the 35 extracted features
# ---------------------------------------------------------------------------

class FeatureBranch(nn.Module):
    """MLP that projects 35 features into the fusion dimension.

    Input:  (batch, n_features)
    Output: (batch, out_dim)
    """
    def __init__(self, n_features=35, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Gated Fusion Block — the core early-fusion mechanism
# ---------------------------------------------------------------------------

class GatedFusionBlock(nn.Module):
    """Learned gated mixing of two representations.

    gate = sigmoid(W @ [feat; sig])
    fused = gate * feat + (1 - gate) * sig

    Then a residual transform produces two new representations
    that feed into the next fusion block.
    """
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        # Shared residual transform after fusion
        self.transform = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        # Separate projections back to each branch
        self.feat_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )
        self.sig_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(self, feat, sig):
        # Gated mixing
        g = self.gate(torch.cat([feat, sig], dim=1))
        fused = g * feat + (1 - g) * sig

        # Residual transform
        fused = fused + self.transform(fused)

        # Project back to separate branches (they diverge again)
        new_feat = feat + self.feat_proj(fused)   # residual from original
        new_sig  = sig  + self.sig_proj(fused)    # residual from original

        return new_feat, new_sig


# ---------------------------------------------------------------------------
# Deep Fusion Network — full model
# ---------------------------------------------------------------------------

class DeepFusionNet(nn.Module):
    """Early fusion network with gated information exchange at every layer.

    Parameters
    ----------
    n_features : int
        Number of extracted features (35).
    n_classes : int
        Number of fault classes.
    n_axes : int
        Number of signal axes (1, 2, or 3).  Missing axes should be zero-padded.
    hidden : int
        Fusion dimension.
    n_fusion_blocks : int
        Number of gated fusion blocks (depth of interaction).
    dropout : float
        Dropout rate.
    """
    def __init__(self, n_features=35, n_classes=5, n_axes=3,
                 hidden=128, n_fusion_blocks=4, dropout=0.2):
        super().__init__()
        self.feature_branch = FeatureBranch(n_features, hidden)
        self.signal_branch = SignalBranch(n_axes, hidden)

        self.fusion_blocks = nn.ModuleList([
            GatedFusionBlock(hidden, dropout) for _ in range(n_fusion_blocks)
        ])

        # Final fusion: combine both branches for classification
        self.final_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 4, n_classes),
        )

    def forward(self, features, signal):
        """
        Parameters
        ----------
        features : (batch, n_features)  — the 35 extracted features (scaled)
        signal   : (batch, n_axes, L)   — raw waveform (zero-pad missing axes)

        Returns
        -------
        logits : (batch, n_classes)
        """
        feat = self.feature_branch(features)   # (B, hidden)
        sig = self.signal_branch(signal)        # (B, hidden)

        # Deep fusion — branches interact at every block
        for fusion_block in self.fusion_blocks:
            feat, sig = fusion_block(feat, sig)

        # Final gated combination
        g = self.final_gate(torch.cat([feat, sig], dim=1))
        combined = g * feat + (1 - g) * sig

        return self.head(combined)

    def get_fusion_weights(self, features, signal):
        """Return gate values for interpretability."""
        feat = self.feature_branch(features)
        sig = self.signal_branch(signal)

        gates = []
        for fusion_block in self.fusion_blocks:
            g = fusion_block.gate(torch.cat([feat, sig], dim=1))
            gates.append(g.detach().cpu())
            feat, sig = fusion_block(feat, sig)

        final_g = self.final_gate(torch.cat([feat, sig], dim=1))
        gates.append(final_g.detach().cpu())
        return gates
