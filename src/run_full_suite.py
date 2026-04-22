#!/usr/bin/env python3
"""
Orchestration: measure always_full reference IoU, train each RL group, greedy-eval,
and record whether mean_iou_ref >= 0.95 * ref.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities import paths  # noqa: E402
from utilities.env_wrappers import build_wrapped_env  # noqa: E402
from utilities.evaluate import rollout  # noqa: E402
from utilities.fish_tracking_env import FishTrackingEnv  # noqa: E402


def _ref_always_full_iou(
    *,
    video: Path,
    weights: Path,
    baseline_npz: Path | None,
    stack_k: int,
    ablation: str,
    device: str,
    seed: int,
    episodes: int,
) -> float:
    from baseline.always_full import always_full_policy

    base = FishTrackingEnv(
        video,
        weights,
        baseline_npz=baseline_npz,
        lambda_cost=0.35,
        max_episode_steps=500,
        imgsz=640,
        device=device,
        random_start=True,
        seed=seed,
    )
    env = build_wrapped_env(base, stack_k=stack_k, ablation=ablation)
    vals: list[float] = []
    for ep in range(episodes):
        st = rollout(env, always_full_policy, seed=seed + ep)
        v = float(st["mean_iou_ref"])
        if not (v != v):  # not nan
            vals.append(v)
    return float(sum(vals) / max(len(vals), 1))


def main() -> None:
    p = argparse.ArgumentParser(description="full suite: ref IoU + train + eval vs 0.95*ref")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--baseline-npz", type=Path, default=None)
    p.add_argument("--stack-k", type=int, default=4)
    p.add_argument("--ablation", default="none")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ref-episodes", type=int, default=5, help="rollouts for always_full reference")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--smoke", action="store_true", help="few train steps (sanity check)")
    p.add_argument("--skip-train", action="store_true", help="only eval existing checkpoints")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    vp = args.video or paths.default_video_path()
    wp = args.weights or paths.default_yolo_weights()
    bp = args.baseline_npz or paths.BASELINE_NPZ
    bp_use = bp if bp.is_file() else None
    if not vp.is_file() or not wp.is_file():
        raise SystemExit("missing video or weights")

    out_dir = args.out_dir or (
        paths.SMOKE_TESTS_DIR if args.smoke else paths.OUTPUTS_DIR / "final_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "suite_summary.csv"

    total_steps = 800 if args.smoke else 50_000
    ref_eps = 2 if args.smoke else int(args.ref_episodes)
    eval_eps = 2 if args.smoke else int(args.eval_episodes)

    ref_iou = _ref_always_full_iou(
        video=vp,
        weights=wp,
        baseline_npz=bp_use,
        stack_k=args.stack_k,
        ablation=args.ablation,
        device=args.device,
        seed=args.seed,
        episodes=ref_eps,
    )
    threshold = 0.95 * ref_iou
    print(
        f"[ref] always_full mean_iou_ref ~ {ref_iou:.4f}  "
        f"pass if agent >= 0.95*ref = {threshold:.4f}",
        flush=True,
    )

    py = sys.executable
    train_py = ROOT / "src" / "train.py"
    eval_py = ROOT / "src" / "evaluate.py"

    jobs: list[tuple[str, list[str]]] = [
        ("g1_vanilla", []),
        ("g2_double_dueling", []),
        ("g3_nstep", ["--n-step", "3"]),
        ("g4_per", []),
        ("g5_softq", []),
        ("g6_c51", []),
        ("g7_gru", []),
    ]

    rows: list[dict[str, object]] = []
    for gid, extra in jobs:
        ckpt = paths.weights_for("RL_agents", gid, "last.pt")
        if not args.skip_train:
            cmd = [
                py,
                str(train_py),
                "--group",
                gid,
                "--video",
                str(vp),
                "--weights",
                str(wp),
                "--stack-k",
                str(args.stack_k),
                "--ablation",
                args.ablation,
                "--total-steps",
                str(total_steps),
                "--learning-starts",
                str(min(100, max(10, total_steps // 8))),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--save",
                str(ckpt),
                "--metrics-csv",
                str(
                    paths.outputs_for("smoke_tests", "RL_agents", gid, "suite_train_metrics.csv")
                    if args.smoke
                    else paths.outputs_for("RL_agents", gid, "suite_train_metrics.csv")
                ),
            ]
            if bp_use is not None:
                cmd.extend(["--baseline-npz", str(bp)])
            cmd.extend(extra)
            print("[run]", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(ROOT), check=True)

        if not ckpt.is_file():
            rows.append(
                {
                    "group": gid,
                    "mean_iou_ref": "",
                    "ref_threshold": f"{threshold:.6f}",
                    "pass_vs_ref": "no_ckpt",
                }
            )
            continue

        ev_csv = out_dir / f"eval_{gid}.csv"
        cmd_e = [
            py,
            str(eval_py),
            "--policy",
            "rl_greedy",
            "--ckpt",
            str(ckpt),
            "--stack-k",
            str(args.stack_k),
            "--ablation",
            args.ablation,
            "--video",
            str(vp),
            "--weights",
            str(wp),
            "--episodes",
            str(eval_eps),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--out-csv",
            str(ev_csv),
        ]
        if bp_use is not None:
            cmd_e.extend(["--baseline-npz", str(bp)])
        print("[eval]", " ".join(cmd_e), flush=True)
        subprocess.run(cmd_e, cwd=str(ROOT), check=True)

        mious: list[float] = []
        if ev_csv.is_file():
            with ev_csv.open(newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    if row.get("mean_iou_ref") not in (None, "", "nan"):
                        try:
                            mious.append(float(row["mean_iou_ref"]))
                        except ValueError:
                            pass
        mean_i = float(sum(mious) / max(len(mious), 1)) if mious else float("nan")
        passed = (not (mean_i != mean_i)) and mean_i >= threshold
        rows.append(
            {
                "group": gid,
                "mean_iou_ref": f"{mean_i:.6f}" if mious else "",
                "ref_threshold": f"{threshold:.6f}",
                "pass_vs_ref": "yes" if passed else "no",
            }
        )
        print(f"[{gid}] mean_iou_ref={mean_i} pass={passed}", flush=True)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["group", "mean_iou_ref", "ref_threshold", "pass_vs_ref"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
