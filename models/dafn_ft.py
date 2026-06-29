#!/usr/bin/env python3
"""DAFN-ImageNet: Uses fine-tuned ResNet50's 3-class logits directly as image features.
The logits (3-dim) are already highly discriminative (97% val acc).
We pass them through a tiny 3→64 FAM, matching the agronomic 10→64 FAM.
This eliminates the 2048→64 random projection instability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FAM(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class MRDS(nn.Module):
    def forward(self, *feats):
        energies = [torch.norm(f, p=2, dim=1, keepdim=True) for f in feats]
        all_e = torch.cat(energies, dim=1)
        w = torch.sigmoid(all_e)
        nw = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        fused = sum(nw[:, i:i+1] * f for i, f in enumerate(feats))
        return fused, nw

class ResFusion(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True), nn.Dropout(0.3),
        )
    def forward(self, x): return x + self.mlp(x)

class DAFN_ImageNet(nn.Module):
    """DAFN using 3-dim logits (fine-tuned ResNet50) + 10-dim agronomic + 6-dim sensor."""
    def __init__(self, logit_dim=3, agronomic_dim=10, sensor_dim=6, hidden=64, num_classes=3, use_sensor=False):
        super().__init__()
        self.use_sensor = use_sensor
        self.fam_img = FAM(logit_dim, hidden)
        self.fam_agr = FAM(agronomic_dim, hidden)
        if use_sensor:
            self.fam_sen = FAM(sensor_dim, hidden)
        self.mrds = MRDS()
        self.res = ResFusion(hidden)
        self.clf = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, logit_feat, agronomic_feat, sensor_feat=None):
        ali = self.fam_img(logit_feat)
        ala = self.fam_agr(agronomic_feat)
        al = [ali, ala]
        if self.use_sensor and sensor_feat is not None:
            al.append(self.fam_sen(sensor_feat))
        fused, w = self.mrds(*al)
        enhanced = self.res(fused)
        return self.clf(enhanced), w

# For backward compatibility - wrapper to accept 2048-dim features via a learned projection
class DAFN_Proj(nn.Module):
    """DAFN with learned projection from 2048 to 64 (used with frozen or fine-tuned ResNet50)."""
    def __init__(self, img_dim=2048, agr_dim=10, sen_dim=6, hidden=64, num_classes=3, use_sensor=False):
        super().__init__()
        self.use_sensor = use_sensor
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU() if hidden > 0 else nn.Identity(),
        )
        self.agr_proj = nn.Sequential(
            nn.Linear(agr_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        if use_sensor:
            self.sen_proj = nn.Sequential(
                nn.Linear(sen_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            )
        self.mrds = MRDS()
        self.res = ResFusion(hidden)
        self.clf = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, img, agr, sen=None):
        pi = F.relu(self.img_proj(img))
        pa = self.agr_proj(agr)
        al = [pi, pa]
        if self.use_sensor and sen is not None:
            al.append(self.sen_proj(sen))
        fused, w = self.mrds(*al)
        return self.clf(self.res(fused)), w
