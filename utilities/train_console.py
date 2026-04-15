"""Small helpers for readable training logs in the terminal."""

from __future__ import annotations

from pathlib import Path


def print_run_header(
    group: str,
    *,
    total_steps: int,
    learning_starts: int,
    state_dim: int,
    stack_k: int,
    ablation: str,
    save_path: Path,
    metrics_csv: Path | None,
    extra_lines: str = "",
) -> None:
    m = str(metrics_csv) if metrics_csv is not None else "(none)"
    tail = f"\n{extra_lines}" if extra_lines else ""
    print(
        f"\n=== [{group}] training ===\n"
        f"  steps={total_steps}  learning_starts={learning_starts}\n"
        f"  state_dim={state_dim}  stack_k={stack_k}  ablation={ablation}\n"
        f"  weights (checkpoint) -> {save_path}\n"
        f"  metrics (csv)        -> {m}{tail}\n",
        flush=True,
    )


def print_episode_done(
    group: str,
    *,
    step: int,
    total_steps: int,
    episodes_done: int,
    ep_return: float,
    ep_len: int,
    buf_size: int,
    epsilon: float,
    lambda_cost: float,
    mean_consistency: float,
) -> None:
    pct = 100.0 * float(step) / float(max(1, total_steps))
    print(
        f"[{group}] step {step}/{total_steps} ({pct:.1f}%) | "
        f"ep#{episodes_done} len={ep_len} return={ep_return:.2f} "
        f"consistency={mean_consistency:.4f} | "
        f"eps={epsilon:.3f} lambda={lambda_cost:.3f} buffer={buf_size}",
        flush=True,
    )


def print_training_done(group: str, *, save_path: Path, final_lambda: float) -> None:
    print(f"[{group}] finished. saved weights to {save_path}  final_lambda={final_lambda:.4f}\n", flush=True)
