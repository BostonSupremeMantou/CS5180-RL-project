from __future__ import annotations

from pathlib import Path

import torch

from utilities.load_rl_policy import load_rl_q_network


def load_policy(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load Q-net; supports g2 dueling checkpoints and newer multi-arch saves."""
    return load_rl_q_network(ckpt_path, device)
