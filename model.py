"""CIFAKE CNN definition and checkpoint loading.

The architecture is identical to the one trained in
notebooks/02_baseline_model.ipynb (634,753 parameters), so the original
``models/cifake_cnn.pth`` state dict loads unchanged.

Class convention (torchvision ImageFolder, alphabetical): FAKE = 0, REAL = 1.
The network outputs a single sigmoid score = P(REAL).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "cifake_cnn.pth"

CLASS_NAMES = ("FAKE", "REAL")  # index 0 / 1


class CIFAKE_CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(self.conv_layers(x))


def load_model(path: Path | str = MODEL_PATH) -> CIFAKE_CNN:
    """Load the trained model on CPU in eval mode.

    ``weights_only=True`` restricts unpickling to tensors, so a tampered
    checkpoint cannot execute code. The file itself is never modified.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    model = CIFAKE_CNN()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model: CIFAKE_CNN, tensor: torch.Tensor) -> dict:
    """Return label, P(REAL) and confidence for a (1, 3, 32, 32) tensor."""
    p_real = float(model(tensor).item())
    label = "REAL" if p_real >= 0.5 else "FAKE"
    confidence = p_real if label == "REAL" else 1.0 - p_real
    return {"label": label, "p_real": p_real, "confidence": confidence}
