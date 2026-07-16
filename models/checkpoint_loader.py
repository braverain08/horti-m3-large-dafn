"""Compatibility wrapper for DAFN-T checkpoint.

The checkpoint (dafn_t_dim128_best.pth) was trained with a different architecture:
- FAM: single Linear+BatchNorm per modality (keys: fam_vis.fc, fam_vis.bn)
- MRDS: learnable gamma scaling per modality (keys: mrds_vis.gamma)
- GRU: hidden_size=80 (not equal to FAM dim=128)
- Classifier: Linear(80 -> 2)
"""
import torch
import torch.nn as nn


class _FAM_V1(nn.Module):
    """Single Linear + BatchNorm, storing submodules as .fc and .bn."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        return self.bn(self.fc(x))


class DAFN_T_Checkpoint(nn.Module):
    """Matches the checkpoint architecture exactly."""
    def __init__(self, window_size=5, fam_dim=128, gru_hidden=80,
                 num_classes=2, use_sensor=False):
        super().__init__()
        self.window_size = window_size
        self.use_sensor = use_sensor

        self.fam_vis = _FAM_V1(2048, fam_dim)
        self.fam_agro = _FAM_V1(10, fam_dim)
        if use_sensor:
            self.fam_sensor = _FAM_V1(6, fam_dim)

        # MRDS with learnable gamma per modality
        # ParameterDict stores gamma as .gamma to match checkpoint keys
        self.mrds_vis = nn.ParameterDict(
            {'gamma': nn.Parameter(torch.tensor(1.0))})
        self.mrds_agro = nn.ParameterDict(
            {'gamma': nn.Parameter(torch.tensor(1.0))})
        self.mrds_sensor = nn.ParameterDict(
            {'gamma': nn.Parameter(torch.tensor(1.0))})

        self.gru = nn.GRU(fam_dim, gru_hidden, batch_first=True)
        self.classifier = nn.Linear(gru_hidden, num_classes)

    def forward(self, image_seq, agronomic_seq, sensor_seq=None):
        B, T = image_seq.shape[:2]
        weight_list = []
        enhanced_list = []

        for t in range(T):
            v = self.fam_vis(image_seq[:, t, :])
            a = self.fam_agro(agronomic_seq[:, t, :])

            e_v = torch.norm(v, p=2, dim=1) * self.mrds_vis['gamma']
            e_a = torch.norm(a, p=2, dim=1) * self.mrds_agro['gamma']
            aligned, energies = [v, a], [e_v, e_a]

            if self.use_sensor and sensor_seq is not None:
                s = self.fam_sensor(sensor_seq[:, t, :])
                e_s = torch.norm(s, p=2, dim=1) * self.mrds_sensor['gamma']
                aligned.append(s)
                energies.append(e_s)

            all_e = torch.stack(energies, dim=1)
            w_norm = torch.sigmoid(all_e)
            w_norm = w_norm / (w_norm.sum(dim=1, keepdim=True) + 1e-8)
            fused = sum(w_norm[:, i:i+1] * a for i, a in enumerate(aligned))
            weight_list.append(w_norm)
            enhanced_list.append(fused)

        enhanced_seq = torch.stack(enhanced_list, dim=1)
        gru_out, _ = self.gru(enhanced_seq)
        last_out = gru_out[:, -1, :]
        logits = self.classifier(last_out)
        return logits, weight_list


def load_dafn_t(path):
    sd = torch.load(path, map_location='cpu')
    model = DAFN_T_Checkpoint(
        window_size=5, fam_dim=128, gru_hidden=80,
        num_classes=2, use_sensor=False,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model
