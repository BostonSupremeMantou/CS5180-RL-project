from __future__ import annotations

import pytest

from agents.RL_agents.registry import RL_GROUP_IDS, get_rl_spec


def test_rl_group_ids_nonempty() -> None:
    assert len(RL_GROUP_IDS) >= 7
    assert "g2_double_dueling" in RL_GROUP_IDS


@pytest.mark.parametrize("gid", RL_GROUP_IDS)
def test_get_rl_spec_each(gid: str) -> None:
    s = get_rl_spec(gid)
    assert s.group_id == gid
    assert s.one_liner


def test_unknown_group_raises() -> None:
    with pytest.raises(KeyError):
        get_rl_spec("not_a_group")
