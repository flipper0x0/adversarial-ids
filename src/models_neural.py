"""Differentiable MLP detector.

Two jobs (proposal section 3.2.4):
  1. White-box target. FGSM/PGD/JSMA need input gradients; tree models have
     none, so the gradient attacks run here.
  2. Surrogate. Adversarial samples crafted on this MLP are transferred to the
     tree models, simulating an attacker who lacks the real model's internals.

No BatchNorm on purpose: BN's train/eval behaviour interacts awkwardly with
single-sample gradient attacks. Plain Linear + ReLU + Dropout keeps the
gradients clean.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import get_logger

log = get_logger("models_neural")


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw logits; loss applies softmax internally


def train_mlp(x_train, y_train, cfg, x_extra=None, y_extra=None) -> MLP:
    """Train the MLP. x_extra/y_extra let adversarial training append samples."""
    device = _device()
    in_dim = x_train.shape[1]
    n_classes = int(np.max(y_train)) + 1
    p = cfg["models"]["mlp"]

    if x_extra is not None and len(x_extra) > 0:
        x_train = np.concatenate([x_train, x_extra], axis=0)
        y_train = np.concatenate([y_train, y_extra], axis=0)

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=p["batch_size"], shuffle=True)

    # Class weights for imbalance.
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    model = MLP(in_dim, n_classes, tuple(p["hidden"]), p["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(weight=weights_t)

    model.train()
    for epoch in range(p["epochs"]):
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        log.info("MLP epoch %d/%d  loss=%.4f", epoch + 1, p["epochs"], total / len(ds))
    model.eval()
    return model


def wrap_art(model: MLP, in_dim: int, n_classes: int, cfg):
    """Wrap the MLP as an ART PyTorchClassifier (clip to [0,1] = scaled range)."""
    from art.estimators.classification import PyTorchClassifier

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["models"]["mlp"]["lr"])
    return PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=opt,
        input_shape=(in_dim,),
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )


def mlp_predict(model: MLP, x: np.ndarray) -> np.ndarray:
    device = _device()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        return logits.argmax(dim=1).cpu().numpy()


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
