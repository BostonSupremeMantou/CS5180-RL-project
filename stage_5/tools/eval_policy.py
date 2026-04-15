"""从 Stage5 / 扩展 architecture 的 ckpt 构建贪心策略（供评估 CSV）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from stage_1.env.fish_tracking_env import STATE_DIM
from stage_1.models.dqn import TrackingDQN, select_action_greedy
from stage_5.models.c51_net import C51TrackingDQN
from stage_5.models.gru_dueling import GRUDuelingDQN

PolicyFn = Callable[[np.ndarray, int], int]


def build_greedy_policy_from_ckpt(ckpt: dict[str, Any], dev: torch.device) -> PolicyFn:
    arch = str(ckpt.get("architecture", "dueling_double"))
    sd = int(ckpt.get("state_dim", 40))
    h = int(ckpt.get("hidden_dim", 256))
    dr = float(ckpt.get("dropout", 0.0))

    if arch == "dueling_double":
        q = TrackingDQN(state_dim=sd, hidden=h, dropout=dr).to(dev)
        q.load_state_dict(ckpt["q_state_dict"])
        q.eval()

        def pol(obs: np.ndarray, _step: int) -> int:
            return select_action_greedy(q, obs, dev)

        return pol

    if arch == "s5_c51":
        q = C51TrackingDQN(state_dim=sd, hidden=h, dropout=dr).to(dev)
        q.load_state_dict(ckpt["q_state_dict"])
        q.eval()

        def pol(obs: np.ndarray, _step: int) -> int:
            t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                return int(q.expected_q(t).argmax(dim=1).item())

        return pol

    if arch == "s5_gru":
        sk = int(ckpt.get("obs_stack_k", 4))
        gh = int(ckpt.get("gru_hidden_dim", h))
        q = GRUDuelingDQN(
            state_dim=sd,
            stack_k=sk,
            raw_dim=STATE_DIM,
            hidden=h,
            gru_hidden=gh,
            dropout=dr,
        ).to(dev)
        q.load_state_dict(ckpt["q_state_dict"])
        q.eval()

        def pol(obs: np.ndarray, _step: int) -> int:
            return select_action_greedy(q, obs, dev)

        return pol

    raise ValueError(f"不支持的 architecture: {arch}")


def eval_custom_arch_row(
    *,
    ckpt_path: Path,
    base_kw: dict,
    seeds: list[int],
    n_episodes: int,
    lambda_from_ckpt: bool,
    dev: torch.device,
    policy_id: str | None = None,
    policy_name: str | None = None,
) -> dict[str, Any]:
    """与 eval_stage2_row 相同协议，但支持 s5_c51 / s5_gru。"""
    from stage_1.env.fish_tracking_env import FishTrackingEnv
    from stage_1.evaluation.baselines import rollout_episode
    from stage_2.env.wrappers import build_stage2_env

    ck = load_ckpt_dict(ckpt_path)
    stack_k = int(ck.get("obs_stack_k", 4))
    abl = str(ck.get("ablation", "none"))
    lam2 = float(base_kw["lambda_cost"])
    if lambda_from_ckpt:
        v = ck.get("lambda_cost_final", ck.get("lambda_cost"))
        if v is not None:
            lam2 = float(v)
    kw2 = {**base_kw, "lambda_cost": lam2}
    pol = build_greedy_policy_from_ckpt(ck, dev)

    rets, cons, teach, costs = [], [], [], []
    for ep in range(n_episodes):
        seed = seeds[ep % len(seeds)]
        be = FishTrackingEnv(**kw2, random_start=True, seed=seed)
        env_ep = build_stage2_env(be, stack_k=stack_k, ablation=abl)
        stats = rollout_episode(env_ep, pol, seed=seed)
        rets.append(stats["return"])
        cons.append(stats["mean_consistency"])
        teach.append(float(stats["mean_iou_teacher"]))
        costs.append(stats["mean_cost"])
    mr = float(np.mean(rets))
    mc = float(np.mean(costs))
    mcons = float(np.mean(cons))
    mteach_vals = [x for x in teach if not np.isnan(x)]
    mteach = float(np.mean(mteach_vals)) if mteach_vals else float("nan")
    pid = policy_id or f"s5_{ckpt_path.stem}"
    pname = policy_name or f"Stage5 {ck.get('architecture', '?')} ({ckpt_path.name})"
    return {
        "policy_id": pid,
        "policy_name": pname,
        "mean_return": mr,
        "mean_consistency": mcons,
        "mean_iou_teacher": mteach,
        "mean_cost": mc,
        "n_episodes": n_episodes,
        "lambda_cost": lam2,
        "ckpt_path": str(ckpt_path),
    }


def load_ckpt_dict(ckpt_path: Path) -> dict[str, Any]:
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")
