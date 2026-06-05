"""Deterministic seeding across random, numpy, and torch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed every RNG Sentinel touches so training runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Favour determinism over the last few % of throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
