#!/usr/bin/env python3
"""
DAFN & DAFN-T: Dual-branch Adaptive Feature Alignment Fusion Network.

Components:
  1. FAM — Feature Alignment Module (projects hetero features to shared 64-d)
  2. MRDS — Modal Reliability Dynamic Scoring (L2-norm based weights)
  3. Res-Fusion — Residual Enhanced Fusion

DAFN  : single-day variant  [image + agronomic + optional sensor → logits]
DAFN-T: temporal variant    [sequence of days → GRU → logits]

All models: ~78K params, ~1.5ms inference/sample (CPU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FAM(nn.Module):
    """Feature Alignment Module: project heterogeneous features to shared space."""
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MRDS(nn.Module):
    """Modal Reliability Dynamic Scoring — L2 norm based adaptive weights."""
    def __init__(self):
        super().__init__()

    def forward(self, *aligned_feats):
        energies = []
        for feat in aligned_feats:
            energy = torch.norm(feat, p=2, dim=1, keepdim=True)
            energies.append(energy)
        all_energies = torch.cat(energies, dim=1)
        weights = torch.sigmoid(all_energies)
        norm_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        fused = sum(w.view(-1, 1) * f for w, f in zip(norm_weights.unbind(dim=1), aligned_feats))
        return fused, norm_weights


class ResFusion(nn.Module):
    """Residual Enhanced Fusion: fused + MLP(fused)."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, fused_feat):
        return fused_feat + self.mlp(fused_feat)


# ==================== DAFN (Single-Day) ====================

class DAFN(nn.Module):
    """Dual-branch Adaptive Feature Alignment Fusion Network (single-day).

    Args:
        image_dim: 2048 (ResNet50)
        agronomic_dim: 10
        sensor_dim: 6 (set 0 to skip)
        hidden_dim: 64
        num_classes: 3 (Healthy / Stress / Other)
        use_sensor: bool
    """
    def __init__(self, image_dim=2048, agronomic_dim=10, sensor_dim=6,
                 hidden_dim=64, num_classes=3, use_sensor=False):
        super().__init__()
        self.use_sensor = use_sensor
        self.hidden_dim = hidden_dim

        self.fam_image = FAM(image_dim, hidden_dim)
        self.fam_agronomic = FAM(agronomic_dim, hidden_dim)
        if use_sensor:
            self.fam_sensor = FAM(sensor_dim, hidden_dim)

        self.mrds = MRDS()
        self.res_fusion = ResFusion(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward_features(self, image_feat, agronomic_feat, sensor_feat=None):
        """Extract enhanced features (without classification head)."""
        aligned_img = self.fam_image(image_feat)
        aligned_agr = self.fam_agronomic(agronomic_feat)
        aligned_list = [aligned_img, aligned_agr]

        if self.use_sensor and sensor_feat is not None:
            aligned_sen = self.fam_sensor(sensor_feat)
            aligned_list.append(aligned_sen)

        fused_feat, weights = self.mrds(*aligned_list)
        enhanced = self.res_fusion(fused_feat)
        return enhanced, weights, aligned_list

    def forward(self, image_feat, agronomic_feat, sensor_feat=None):
        enhanced, weights, _ = self.forward_features(image_feat, agronomic_feat, sensor_feat)
        logits = self.classifier(enhanced)
        return logits, weights


# ==================== DAFN-T (Temporal) ====================

class DAFN_T(nn.Module):
    """DAFN with temporal GRU extension.

    Processes a sliding window of N consecutive days:
      1. DAFN forward_features per timestep (shared weights)
      2. GRU across timesteps
      3. Classification from GRU final output

    Args:
        window_size: number of consecutive days (default: 5)
        image_dim: 2048
        agronomic_dim: 10
        sensor_dim: 6
        hidden_dim: 64
        num_classes: 3
        use_sensor: bool
    """
    def __init__(self, window_size=5, **dafn_kwargs):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = dafn_kwargs.get('hidden_dim', 64)

        # Build DAFN feature extractor (no classifier head)
        self.fam_image = FAM(dafn_kwargs.get('image_dim', 2048), self.hidden_dim)
        self.fam_agronomic = FAM(dafn_kwargs.get('agronomic_dim', 10), self.hidden_dim)
        self.use_sensor = dafn_kwargs.get('use_sensor', False)
        if self.use_sensor:
            self.fam_sensor = FAM(dafn_kwargs.get('sensor_dim', 6), self.hidden_dim)

        self.mrds = MRDS()
        self.res_fusion = ResFusion(self.hidden_dim)

        # Temporal GRU
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim,
                          num_layers=1, batch_first=True, bidirectional=False)

        # Classifier on top of GRU output
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, dafn_kwargs.get('num_classes', 3)),
        )

    def forward(self, image_seq, agronomic_seq, sensor_seq=None):
        """
        Args:
            image_seq: [B, T, 2048] where T = window_size
            agronomic_seq: [B, T, 10]
            sensor_seq: [B, T, 6] or None
        Returns:
            logits: [B, num_classes]
            weights_list: list of T weight tensors [B, num_modalities]
        """
        B, T = image_seq.shape[:2]
        enhanced_list = []
        weight_list = []

        # Process each timestep with shared DAFN components
        for t in range(T):
            img_t = image_seq[:, t, :]
            agr_t = agronomic_seq[:, t, :]

            aligned_img = self.fam_image(img_t)
            aligned_agr = self.fam_agronomic(agr_t)
            aligned = [aligned_img, aligned_agr]

            if self.use_sensor and sensor_seq is not None:
                sen_t = sensor_seq[:, t, :]
                aligned_sen = self.fam_sensor(sen_t)
                aligned.append(aligned_sen)

            fused_t, w_t = self.mrds(*aligned)
            enhanced_t = self.res_fusion(fused_t)
            enhanced_list.append(enhanced_t)
            weight_list.append(w_t)

        # Stack and run GRU
        enhanced_seq = torch.stack(enhanced_list, dim=1)  # [B, T, D]
        gru_out, _ = self.gru(enhanced_seq)                # [B, T, D]
        last_out = gru_out[:, -1, :]                       # [B, D]

        logits = self.classifier(last_out)
        return logits, weight_list


# ==================== Baseline Models ====================

class ImageOnlyClassifier(nn.Module):
    """2048 → 256 → 128 → num_classes."""
    def __init__(self, input_dim=2048, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class AgronomicOnlyClassifier(nn.Module):
    """10 → 32 → 16 → num_classes."""
    def __init__(self, input_dim=10, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleFusion(nn.Module):
    """Concat + MLP."""
    def __init__(self, image_dim=2048, agronomic_dim=10, hidden_dim=64, num_classes=3):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.agr_proj = nn.Linear(agronomic_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_feat, agronomic_feat):
        img = F.relu(self.image_proj(image_feat))
        agr = F.relu(self.agr_proj(agronomic_feat))
        fused = torch.cat([img, agr], dim=1)
        return self.classifier(fused)


class ConcatCBAM(nn.Module):
    """Early Concat + Channel Attention."""
    def __init__(self, image_dim=2048, agronomic_dim=10, hidden_dim=128, num_classes=3):
        super().__init__()
        self.proj = nn.Linear(image_dim + agronomic_dim, hidden_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, image_feat, agronomic_feat):
        x = torch.cat([image_feat, agronomic_feat], dim=1)
        x = F.relu(self.proj(x))
        x_t = x.unsqueeze(-1)
        weight = self.gap(x_t).squeeze(-1)
        weight = F.relu(self.fc1(weight))
        weight = torch.sigmoid(self.fc2(weight))
        x = x * weight
        return self.classifier(x)


class GatedFusion(nn.Module):
    """Gated Multimodal Fusion."""
    def __init__(self, image_dim=2048, agronomic_dim=10, hidden_dim=64, num_classes=3):
        super().__init__()
        self.img_proj = nn.Linear(image_dim, hidden_dim)
        self.agr_proj = nn.Linear(agronomic_dim, hidden_dim)
        self.gate_img = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_agr = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, image_feat, agronomic_feat):
        img_h = F.relu(self.img_proj(image_feat))
        agr_h = F.relu(self.agr_proj(agronomic_feat))
        cat = torch.cat([img_h, agr_h], dim=1)
        g_i = torch.sigmoid(self.gate_img(cat))
        g_a = torch.sigmoid(self.gate_agr(cat))
        fused = g_i * img_h + g_a * agr_h
        return self.classifier(fused)


class CrossAttentionFusion(nn.Module):
    """Single-head cross-attention fusion."""
    def __init__(self, image_dim=2048, agronomic_dim=10, hidden_dim=64, num_classes=3):
        super().__init__()
        self.img_proj = nn.Linear(image_dim, hidden_dim)
        self.agr_proj = nn.Linear(agronomic_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, image_feat, agronomic_feat):
        img_h = F.relu(self.img_proj(image_feat)).unsqueeze(1)
        agr_h = F.relu(self.agr_proj(agronomic_feat)).unsqueeze(1)
        attn_out, _ = self.cross_attn(img_h, agr_h, agr_h)
        fused = self.norm(img_h + attn_out).squeeze(1)
        return self.classifier(fused)


class AlignmentLoss(nn.Module):
    """Encourages similar feature distributions across modalities (Section 4.6)."""
    def forward(self, aligned_list):
        if len(aligned_list) < 2:
            return torch.tensor(0.0, device=aligned_list[0].device)
        loss = torch.tensor(0.0, device=aligned_list[0].device)
        count = 0
        for i in range(len(aligned_list)):
            for j in range(i + 1, len(aligned_list)):
                loss += F.mse_loss(aligned_list[i].mean(dim=0), aligned_list[j].mean(dim=0))
                loss += F.mse_loss(aligned_list[i].std(dim=0), aligned_list[j].std(dim=0))
                count += 1
        return loss / count
