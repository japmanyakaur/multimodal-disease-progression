"""3D MRI branch: simulated T1-like volumes, a small 3D ResNet encoder and Grad-CAM.

Volumes are rendered from each scan's regional measures (hippocampus, entorhinal cortex,
ventricles, whole brain, ICV) so the image and the derived measures agree, then degraded with
scanner artefacts that depend on the scan's quality rating (noise, blur, motion ghosting,
bias field, small misregistration). The CNN therefore has to *measure* atrophy from pixels.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from scipy import ndimage

GRID_HI, GRID = 64, 32
_lin = np.linspace(-1, 1, GRID_HI, dtype=np.float32)
_X, _Y, _Z = np.meshgrid(_lin, _lin, _lin, indexing="ij")           # x: left-right, y: back-front, z: down-up
CSF, GM, WM, HIPP = 0.15, 0.55, 0.85, 0.58
NOISE = {1: 0.03, 2: 0.05, 3: 0.09}

# Anatomical landmarks used for the Grad-CAM plausibility check (centres in normalised coordinates)
HIPPO_CENTRES = [(-0.34, -0.05, -0.30), (0.34, -0.05, -0.30)]
VENT_CENTRES = [(-0.13, 0.05, 0.12), (0.13, 0.05, 0.12)]


def _ell(c, r):
    return ((_X - c[0]) / r[0]) ** 2 + ((_Y - c[1]) / r[1]) ** 2 + ((_Z - c[2]) / r[2]) ** 2 <= 1.0


def render(hipp, ent, vent, brain, icv, qc, seed):
    """Render one 32^3 T1-like volume from regional measures and a quality rating (1 good .. 3 poor)."""
    rng = np.random.default_rng(seed)
    s = (icv / 1.45e6) ** (1 / 3)
    hr = np.clip(hipp / icv * 1e3 / 5.2, 0.4, 1.3)               # hippocampal size relative to a healthy brain
    er = np.clip(ent / icv * 1e3 / 2.7, 0.4, 1.3)
    vr = np.clip(vent / icv * 1e2 / 1.5, 0.5, 5.0)
    br = np.clip(brain / icv / 0.74, 0.8, 1.1)
    r0 = np.array([0.78, 0.92, 0.80]) * s

    v = np.zeros((GRID_HI,) * 3, np.float32)
    v[_ell((0, 0, 0), r0)] = CSF                                  # intracranial space
    rb = r0 * br ** (1 / 3)
    v[_ell((0, 0, 0), rb)] = GM                                   # cortex
    v[_ell((0, 0, 0), rb * (1 - 0.2 * br))] = WM                  # white matter
    f_v = vr ** (1 / 3)
    for c in VENT_CENTRES:
        v[_ell(np.array(c) * s, np.array([0.07, 0.30, 0.11]) * s * f_v)] = CSF
    f_h, f_e = hr ** (1 / 3), er ** (1 / 3)
    for c in HIPPO_CENTRES:
        side = np.sign(c[0])
        horn = np.array([0.03 + 0.20 * (1.1 - hr), 0.22, 0.03 + 0.16 * (1.1 - hr)]).clip(0.02) * s
        v[_ell(np.array([c[0] + side * 0.04, c[1], c[2] + 0.06]) * s, horn)] = CSF   # temporal horn widens
        v[_ell(np.array(c) * s, np.array([0.10, 0.24, 0.10]) * s * f_h)] = HIPP
        v[_ell(np.array([c[0] - side * 0.03, 0.14, -0.36]) * s, np.array([0.06, 0.08, 0.05]) * s * f_e)] = GM

    v = v.reshape(GRID, 2, GRID, 2, GRID, 2).mean(axis=(1, 3, 5))  # partial-volume downsampling
    lin = np.linspace(-1, 1, GRID, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    a = rng.normal(0, 0.06, 3)
    v *= 1 + a[0] * gx + a[1] * gy + a[2] * gz                    # bias field
    v = ndimage.shift(v, rng.normal(0, 0.5, 3), order=1)          # residual misregistration
    if qc >= 2:
        v = ndimage.gaussian_filter1d(v, 0.6 if qc == 2 else 1.0, axis=1)
    if qc == 3:
        v = v + 0.15 * np.roll(v, 5, axis=1)                      # motion ghost
    sd = NOISE[int(qc)]
    v = np.sqrt((v + rng.normal(0, sd, v.shape)) ** 2 + rng.normal(0, sd, v.shape) ** 2)   # Rician noise
    return v.astype(np.float16)


def normalise(vol):
    """Z-score inside the head (deterministic, no fitted statistics)."""
    v = vol.astype(np.float32)
    mask = v > 0.1
    return ((v - v[mask].mean()) / (v[mask].std() + 1e-6) * mask).astype(np.float32)


def roi_masks():
    """Boolean 32^3 masks of hippocampal/temporal and ventricular regions (for Grad-CAM plausibility)."""
    lin = np.linspace(-1, 1, GRID, dtype=np.float32)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")

    def ell(c, r):
        return ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2 <= 1.0

    hippo = np.zeros_like(x, bool)
    vent = np.zeros_like(x, bool)
    for c in HIPPO_CENTRES:
        hippo |= ell(c, (0.18, 0.32, 0.18))
    for c in VENT_CENTRES:
        vent |= ell(c, (0.14, 0.40, 0.18))
    head = ell((0, 0, 0), (0.80, 0.94, 0.82))
    return hippo, vent, head


class ResBlock3d(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm3d(cout)
        self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        return torch.relu(self.b2(self.c2(torch.relu(self.b1(self.c1(x))))) + self.skip(x))


class MRICNN(nn.Module):
    """Compact 3D ResNet: 32^3 -> 16^3 -> 8^3 -> 4^3, global average pooling -> 64-d embedding."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 16, 3, 2, 1, bias=False), nn.BatchNorm3d(16), nn.ReLU())
        self.block1 = ResBlock3d(16, 32, 2)
        self.block2 = ResBlock3d(32, 64, 2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 1))

    def features(self, x):
        f1 = self.block1(self.stem(x))
        f2 = self.block2(f1)
        return f1, f2

    def forward(self, x):
        _, f2 = self.features(x)
        emb = f2.mean(dim=(2, 3, 4))
        return self.head(emb).squeeze(-1), emb, f2


def augment(x, gen):
    """Small shifts and intensity changes; no left-right flips (atrophy can be asymmetric)."""
    shifts = torch.randint(-2, 3, (3,), generator=gen).tolist()
    x = torch.roll(x, shifts=shifts, dims=(2, 3, 4))
    scale = 1 + 0.1 * (torch.rand((x.shape[0], 1, 1, 1, 1), generator=gen) - 0.5)
    return x * scale + 0.03 * torch.randn(x.shape, generator=gen)


def grad_cam(model, x, layer="block1"):
    """Grad-CAM at the 8^3 (block1) or 4^3 (block2) feature map, upsampled to 32^3 and scaled to [0, 1]."""
    model.eval()
    x = x.clone().requires_grad_(False)
    f1, f2 = model.features(x)
    feat = f1 if layer == "block1" else f2
    feat.retain_grad()
    emb = f2.mean(dim=(2, 3, 4))
    model.head(emb).sum().backward()
    w = feat.grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((w * feat).sum(1, keepdim=True))
    cam = Fnn.interpolate(cam, size=(GRID,) * 3, mode="trilinear", align_corners=False)[:, 0]
    cam = cam / (cam.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    return cam.detach().numpy()
