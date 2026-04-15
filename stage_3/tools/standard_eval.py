"""与 compare_stage2 中 Stage2 段一致：对多个 ckpt 做贪心评估并写 CSV。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

def eval_stage2_row(
    *,
    ckpt_path: Path,
    base_kw: dict,
    seeds: list[int],
    n_episodes: int,
    lambda_from_ckpt: bool,
    dev: torch.device,
) -> dict[str, Any]:
    from stage_1.env.fish_tracking_env import FishTrackingEnv
    from stage_1.evaluation.baselines import rollout_episode
    from stage_1.models.dqn import load_q_from_checkpoint, select_action_greedy
    from stage_2.env.wrappers import build_stage2_env

    try:
        ckpt2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt2 = torch.load(ckpt_path, map_location="cpu")
    stack_k = int(ckpt2.get("obs_stack_k", 4))
    abl = str(ckpt2.get("ablation", "none"))
    lam2 = float(base_kw["lambda_cost"])
    if lambda_from_ckpt:
        v = ckpt2.get("lambda_cost_final", ckpt2.get("lambda_cost"))
        if v is not None:
            lam2 = float(v)
    kw2 = {**base_kw, "lambda_cost": lam2}
    q2 = load_q_from_checkpoint(ckpt2, dev)

    def pol2(obs: np.ndarray, _s: int) -> int:
        return select_action_greedy(q2, obs, dev)

    rets, cons, teach, costs = [], [], [], []
    for ep in range(n_episodes):
        seed = seeds[ep % len(seeds)]
        be = FishTrackingEnv(**kw2, random_start=True, seed=seed)
        env_ep = build_stage2_env(be, stack_k=stack_k, ablation=abl)
        stats = rollout_episode(env_ep, pol2, seed=seed)
        rets.append(stats["return"])
        cons.append(stats["mean_consistency"])
        teach.append(float(stats["mean_iou_teacher"]))
        costs.append(stats["mean_cost"])
    mr = float(np.mean(rets))
    mc = float(np.mean(costs))
    mcons = float(np.mean(cons))
    mteach_vals = [x for x in teach if not np.isnan(x)]
    mteach = float(np.mean(mteach_vals)) if mteach_vals else float("nan")
    return {
        "policy_id": f"dqn_stage2_sk{stack_k}",
        "policy_name": f"DQN Stage2 stack_k={stack_k} ablation={abl} ({ckpt_path.name})",
        "mean_return": mr,
        "mean_consistency": mcons,
        "mean_iou_teacher": mteach,
        "mean_cost": mc,
        "n_episodes": n_episodes,
        "lambda_cost": lam2,
        "ckpt_path": str(ckpt_path),
    }


def baseline_rows(
    *,
    base_kw: dict,
    seeds: list[int],
    n_episodes: int,
) -> list[dict[str, Any]]:
    from stage_1.evaluation.baselines import (
        policy_always_full,
        policy_optical_flow_only,
        policy_periodic,
    )
    from stage_1.evaluation.run_eval import evaluate_policy

    rows: list[dict[str, Any]] = []
    r = evaluate_policy(
        policy_id="always_full",
        policy_name="Baseline: always FULL_DETECT",
        env_kwargs=base_kw,
        policy_fn=policy_always_full,
        n_episodes=n_episodes,
        seeds=seeds,
        random_start=False,
    )
    rows.append({**r, "lambda_cost": float(base_kw["lambda_cost"]), "ckpt_path": ""})
    r = evaluate_policy(
        policy_id="periodic_5",
        policy_name="Baseline: periodic FULL every 5 (else REUSE)",
        env_kwargs=base_kw,
        policy_fn=policy_periodic(5),
        n_episodes=n_episodes,
        seeds=seeds,
        random_start=False,
    )
    rows.append({**r, "lambda_cost": float(base_kw["lambda_cost"]), "ckpt_path": ""})
    r = evaluate_policy(
        policy_id="flow_only",
        policy_name="Baseline: optical flow only (LIGHT_UPDATE)",
        env_kwargs=base_kw,
        policy_fn=policy_optical_flow_only,
        n_episodes=n_episodes,
        seeds=seeds,
        random_start=False,
    )
    rows.append({**r, "lambda_cost": float(base_kw["lambda_cost"]), "ckpt_path": ""})
    return rows


def write_standard_eval_csv(
    *,
    out_csv: Path,
    rows: list[dict[str, Any]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = [
        "policy_id",
        "policy_name",
        "mean_return",
        "mean_consistency",
        "mean_iou_teacher",
        "mean_cost",
        "n_episodes",
        "lambda_cost",
        "ckpt_path",
    ]
    for k in sorted(keys):
        if k not in fieldnames:
            fieldnames.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ok] 标准评估表 -> {out_csv}")


def run_standard_eval(
    *,
    checkpoint_glob: str,
    out_csv: Path,
    video_path: Path | None,
    teacher_npz: Path | None,
    baseline_lambda: float,
    n_episodes: int,
    lambda_from_ckpt: bool,
    device: str,
    use_default_teacher: bool = True,
) -> None:
    from stage_1.utils.paths import DEFAULT_TEACHER_NPZ, VIDEO_PATH, WEIGHTS_PATH

    vp = Path(video_path) if video_path is not None else VIDEO_PATH
    if not vp.is_file():
        raise SystemExit(f"找不到视频: {vp}")
    tn: Path | None = None
    if teacher_npz is not None:
        tnp = Path(teacher_npz)
        if tnp.is_file():
            tn = tnp
    elif use_default_teacher and DEFAULT_TEACHER_NPZ.is_file():
        tn = DEFAULT_TEACHER_NPZ

    base_kw = dict(
        video_path=vp,
        yolo_weights=WEIGHTS_PATH,
        teacher_npz=tn,
        lambda_cost=baseline_lambda,
        max_episode_steps=200,
        imgsz=640,
        device=device,
    )
    seeds = [0, 1, 2, 3, 4][: max(1, n_episodes)]
    rows = baseline_rows(base_kw=base_kw, seeds=seeds, n_episodes=n_episodes)

    ckpt_dir = Path(checkpoint_glob).parent
    pat = Path(checkpoint_glob).name
    paths = sorted(ckpt_dir.glob(pat))
    if not paths:
        raise SystemExit(f"未找到 checkpoint: {checkpoint_glob}")
    dev = torch.device("cpu")
    for ck in paths:
        rows.append(
            eval_stage2_row(
                ckpt_path=ck,
                base_kw=base_kw,
                seeds=seeds,
                n_episodes=n_episodes,
                lambda_from_ckpt=lambda_from_ckpt,
                dev=dev,
            )
        )
    write_standard_eval_csv(out_csv=out_csv, rows=rows)
