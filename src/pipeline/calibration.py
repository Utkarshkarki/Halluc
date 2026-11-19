# src/pipeline/calibration.py
"""Simple temperature‑scaling calibration for NLI logits.

The calibration step is optional; the pipeline can operate without it.
When enabled, a :class:`CalibrationModule` learns a single scalar
``temperature`` that smooths the raw NLI logits before converting them to
probabilities.  The implementation is lightweight and has no external
dependencies beyond ``torch`` (which is already used by the NLI detector).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class CalibrationModule:
    """Temperature‑scaling calibrator for NLI logits.

    Parameters
    ----------
    enabled: bool, default ``False``
        Whether calibration should be applied.  If ``False`` the
        :meth:`apply` method returns the input logits unchanged.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.temperature: float | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200, lr: float = 0.01) -> None:
        """Learn a temperature that maximises the likelihood of ``labels``.

        Parameters
        ----------
        logits: np.ndarray, shape ``(n_samples, n_classes)``
            Raw NLI logits produced by the model.
        labels: np.ndarray, shape ``(n_samples,)``
            Integer class indices (0‑based) matching the order of logits.
        max_iter: int, default ``200``
            Number of gradient‑descent steps.
        lr: float, default ``0.01``
            Learning rate for the temperature parameter.
        """
        if not self.enabled:
            return
        if logits.shape[0] == 0:
            raise ValueError("Empty logits array – cannot fit calibration.")
        device = torch.device("cpu")
        t = torch.nn.Parameter(torch.tensor(1.0, device=device))  # temperature
        optimizer = torch.optim.Adam([t], lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        logits_t = torch.from_numpy(logits).float().to(device)
        labels_t = torch.from_numpy(labels).long().to(device)
        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled = logits_t / t.clamp_min(1e-6)
            loss = loss_fn(scaled, labels_t)
            loss.backward()
            optimizer.step()
        self.temperature = float(t.item())

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply the learned temperature to ``logits``.

        Returns the calibrated logits (still raw, not softmaxed).  If the
        module is disabled or not fitted yet, the original logits are
        returned unchanged.
        """
        if not self.enabled or self.temperature is None:
            return logits
        calibrated = logits / self.temperature
        return calibrated

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serialise the temperature to ``path`` using ``pickle``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"temperature": self.temperature, "enabled": self.enabled}, f)

    def load(self, path: Path) -> None:
        """Load a previously saved calibration state.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.temperature = data.get("temperature")
        self.enabled = data.get("enabled", False)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def is_fitted(self) -> bool:
        """Return ``True`` if a temperature has been learned.
        """
        return self.temperature is not None

    def __repr__(self) -> str:  # pragma: no cover – debugging helper
        return f"CalibrationModule(enabled={self.enabled}, temperature={self.temperature})"


__all__ = ["CalibrationModule"]
