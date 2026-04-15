"""lookup RL 'group' specs — hot-swap entry point for src/*.py."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

# keep ids short — matches folder names under weights/RL_agents and outputs/RL_agents
RL_GROUP_IDS: tuple[str, ...] = (
    "g1_vanilla",
    "g2_double_dueling",
    "g3_nstep",
    "g4_per",
    "g5_softq",
    "g6_c51",
    "g7_gru",
)


@dataclass
class RLSpec:
    group_id: str
    one_liner: str
    legacy_train_hint: str  # where to copy code from until unified trainer exists


_SPECS: dict[str, RLSpec] = {
    "g1_vanilla": RLSpec(
        "g1_vanilla",
        "plain MLP DQN (stacked state; not legacy 8-D ckpts)",
        "agents/RL_agents/g1_vanilla/train.py",
    ),
    "g2_double_dueling": RLSpec(
        "g2_double_dueling",
        "Dueling + Double DQN + uniform replay + 1-step",
        "agents/RL_agents/g2_double_dueling/train.py + utilities/*",
    ),
    "g3_nstep": RLSpec(
        "g3_nstep",
        "same as g2 but n-step replay bridge",
        "old_versions/stage_2/training/train_stage2.py --n-step N",
    ),
    "g4_per": RLSpec(
        "g4_per",
        "prioritized replay + IS weights + Double DQN",
        "old_versions/stage_5/training/train_stage5.py --stage5-mode per",
    ),
    "g5_softq": RLSpec(
        "g5_softq",
        "soft entropy backup on target net",
        "old_versions/stage_5/training/train_stage5.py --stage5-mode softq",
    ),
    "g6_c51": RLSpec(
        "g6_c51",
        "c51 distribution + projection loss",
        "old_versions/stage_5/training/train_stage5.py --stage5-mode c51",
    ),
    "g7_gru": RLSpec(
        "g7_gru",
        "GRU over stacked raw frames then dueling head",
        "old_versions/stage_5/training/train_stage5.py --stage5-mode gru",
    ),
}


def get_rl_spec(group_id: str) -> RLSpec:
    if group_id not in _SPECS:
        raise KeyError(f"unknown rl group {group_id!r}, pick one of {RL_GROUP_IDS}")
    return _SPECS[group_id]


def load_optional_hook(group_id: str, attr: str) -> Any | None:
    """if a group ships a real hook, it lives in agents.RL_agents.<pkg>.spec."""
    try:
        mod = import_module(f"agents.RL_agents.{group_id}.spec")
    except ImportError:
        return None
    return getattr(mod, attr, None)
