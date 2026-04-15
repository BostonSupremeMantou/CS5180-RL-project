"""g1: Vanilla MLP DQN + uniform replay (same stacked obs as other groups)."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utilities.env_wrappers import build_wrapped_env, stacked_state_dim
from utilities.fish_tracking_env import FishTrackingEnv
from utilities.nn.dueling import select_action_epsilon_greedy
from utilities.nn.vanilla import VanillaDQN
from utilities.replay_buffer import ReplayBuffer
from utilities.torch_train import augment_obs_batch, polyak_update
from utilities.train_console import print_episode_done, print_run_header, print_training_done
from utilities.train_early_stop import RefIoUEarlyStopTracker


def train_g1(
    *,
    video_path: Path,
    yolo_weights: Path,
    baseline_npz: Path | None,
    stack_k: int = 4,
    ablation: str = "none",
    total_steps: int = 50_000,
    learning_starts: int | None = None,
    train_freq: int = 4,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 3e-4,
    lr_min: float = 1e-5,
    weight_decay: float = 0.0,
    use_cosine_lr: bool = True,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 40_000,
    target_update_every: int = 1_000,
    target_tau: float = 0.0,
    replay_capacity: int = 100_000,
    lambda_cost: float = 0.35,
    max_episode_steps: int = 500,
    imgsz: int = 640,
    device_s: str = "cpu",
    seed: int = 0,
    save_path: Path,
    metrics_csv: Path | None = None,
    metrics_ma_episodes: int = 20,
    fixed_lambda: bool = False,
    target_mean_cost: float = 0.45,
    lambda_lr: float = 0.03,
    lambda_initial: float = 0.45,
    lambda_min: float = 0.25,
    lambda_max: float = 1.5,
    huber_beta: float = 1.0,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    obs_noise_std: float = 0.0,
    policy_obs_noise_std: float = 0.0,
    grad_updates: int = 1,
    log_every: int = 100,
    ref_iou_early_stop: RefIoUEarlyStopTracker | None = None,
) -> None:
    dev = torch.device("cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    ls = learning_starts if learning_starts is not None else min(5_000, max(100, total_steps // 4))
    lam = float(lambda_initial if not fixed_lambda else lambda_cost)
    base = FishTrackingEnv(
        video_path,
        yolo_weights,
        baseline_npz=baseline_npz,
        lambda_cost=lam,
        max_episode_steps=max_episode_steps,
        imgsz=imgsz,
        device=device_s,
        random_start=True,
        seed=seed,
    )
    env = build_wrapped_env(base, stack_k=stack_k, ablation=ablation)
    state_dim = stacked_state_dim(stack_k)

    q = VanillaDQN(state_dim=state_dim, hidden=hidden_dim, dropout=dropout).to(dev)
    target = VanillaDQN(state_dim=state_dim, hidden=hidden_dim, dropout=dropout).to(dev)
    target.load_state_dict(q.state_dict())
    target.train()
    q.train()

    opt = torch.optim.Adam(q.parameters(), lr=lr, weight_decay=weight_decay)
    gu = max(1, int(grad_updates))
    est_grad_steps = max(1, (total_steps - max(ls, 0)) // max(train_freq, 1) * gu)
    sched: torch.optim.lr_scheduler.LRScheduler | None = None
    if use_cosine_lr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=est_grad_steps, eta_min=lr_min
        )

    buf = ReplayBuffer(replay_capacity, state_dim=state_dim, rng=rng)
    print_run_header(
        "g1_vanilla",
        total_steps=total_steps,
        learning_starts=ls,
        state_dim=state_dim,
        stack_k=stack_k,
        ablation=ablation,
        save_path=save_path,
        metrics_csv=metrics_csv,
    )
    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    ep_len = 0
    ep_actions: list[int] = []
    step = 0
    pbar = tqdm(total=total_steps, desc="g1-vanilla")
    losses_since_log: list[float] = []
    td_abs_since_log: list[float] = []
    grad_norms_since_log: list[float] = []
    ep_returns_ma = deque(maxlen=max(1, metrics_ma_episodes))
    ep_ious_ma = deque(maxlen=max(1, metrics_ma_episodes))
    ep_teacher_ma = deque(maxlen=max(1, metrics_ma_episodes))
    ep_comps_ma = deque(maxlen=max(1, metrics_ma_episodes))
    ep_full_frac_ma = deque(maxlen=max(1, metrics_ma_episodes))
    episodes_done = 0
    use_soft_target = target_tau > 0.0

    csv_file = None
    csv_writer = None
    if metrics_csv is not None:
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = metrics_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "step",
                "epsilon",
                "lambda_cost",
                "lr",
                "buffer_size",
                "loss_mean",
                "td_abs_mean",
                "grad_norm_mean",
                "ep_ret_ma",
                "ep_consistency_ma",
                "ep_teacher_iou_ma",
                "ep_comp_ma",
                "ep_full_frac_ma",
                "episodes_done",
            ]
        )

    def log_metrics() -> None:
        nonlocal losses_since_log, td_abs_since_log, grad_norms_since_log
        buf_n = len(buf)
        loss_m = float(np.mean(losses_since_log)) if losses_since_log else float("nan")
        td_m = float(np.mean(td_abs_since_log)) if td_abs_since_log else float("nan")
        gn_m = float(np.mean(grad_norms_since_log)) if grad_norms_since_log else float("nan")
        r_ma = float(np.mean(ep_returns_ma)) if ep_returns_ma else float("nan")
        i_ma = float(np.mean(ep_ious_ma)) if ep_ious_ma else float("nan")
        t_ma = (
            float(np.nanmean(np.asarray(list(ep_teacher_ma), dtype=np.float64)))
            if ep_teacher_ma
            else float("nan")
        )
        c_ma = float(np.mean(ep_comps_ma)) if ep_comps_ma else float("nan")
        f_ma = float(np.mean(ep_full_frac_ma)) if ep_full_frac_ma else float("nan")
        lam_s = env.lambda_cost
        lr_now = float(opt.param_groups[0]["lr"])
        t_msg = f" baseline_iou={t_ma:.4f}" if ep_teacher_ma and not np.isnan(t_ma) else ""
        tqdm.write(
            f"[metrics] step={step} eps={eps:.4f} lambda={lam_s:.4f} lr={lr_now:.2e} buf={buf_n} | "
            f"loss={loss_m:.5f} td={td_m:.5f} |grad|={gn_m:.4f} | "
            f"ep_ma{metrics_ma_episodes}: ret={r_ma:.2f} consistency={i_ma:.4f}{t_msg} "
            f"compute={c_ma:.2f} full%={f_ma:.3f} | episodes={episodes_done}"
        )
        if csv_writer is not None:
            csv_writer.writerow(
                [
                    step,
                    f"{eps:.6f}",
                    f"{lam_s:.6f}",
                    f"{lr_now:.8f}",
                    buf_n,
                    f"{loss_m:.8f}" if not np.isnan(loss_m) else "",
                    f"{td_m:.8f}" if not np.isnan(td_m) else "",
                    f"{gn_m:.8f}" if not np.isnan(gn_m) else "",
                    f"{r_ma:.8f}" if not np.isnan(r_ma) else "",
                    f"{i_ma:.8f}" if not np.isnan(i_ma) else "",
                    f"{t_ma:.8f}" if ep_teacher_ma and not np.isnan(t_ma) else "",
                    f"{c_ma:.8f}" if not np.isnan(c_ma) else "",
                    f"{f_ma:.8f}" if not np.isnan(f_ma) else "",
                    episodes_done,
                ]
            )
            csv_file.flush()
        losses_since_log.clear()
        td_abs_since_log.clear()
        grad_norms_since_log.clear()

    while step < total_steps:
        eps = float(
            epsilon_end
            + (epsilon_start - epsilon_end)
            * max(0.0, 1.0 - step / float(max(1, epsilon_decay_steps)))
        )
        obs_for_policy = obs.astype(np.float32)
        if policy_obs_noise_std > 0:
            obs_for_policy = (
                obs_for_policy
                + rng.normal(0.0, policy_obs_noise_std, size=obs_for_policy.shape).astype(
                    np.float32
                )
            )
        a = select_action_epsilon_greedy(q, obs_for_policy, eps, 3, dev, rng)
        next_obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        buf.push(obs, a, r, next_obs, done)
        ep_return += float(r)
        ep_len += 1
        ep_actions.append(int(a))
        obs = next_obs
        step += 1
        pbar.update(1)

        if done:
            episodes_done += 1
            n_full = sum(1 for x in ep_actions if x == 0)
            full_frac = n_full / max(ep_len, 1)
            ep_returns_ma.append(ep_return)
            ep_ious_ma.append(
                float(info.get("episode_mean_consistency", info.get("episode_mean_iou", 0.0)))
            )
            ep_teacher_ma.append(float(info.get("episode_mean_iou_teacher", float("nan"))))
            ep_comps_ma.append(float(info.get("episode_compute", 0.0)))
            ep_full_frac_ma.append(full_frac)
            if not fixed_lambda:
                mean_cost = float(info.get("episode_mean_cost", 0.0))
                lam = lam + lambda_lr * (mean_cost - target_mean_cost)
                lam = float(np.clip(lam, lambda_min, lambda_max))
                env.lambda_cost = lam
            mean_cons = float(
                info.get("episode_mean_consistency", info.get("episode_mean_iou", 0.0))
            )
            print_episode_done(
                "g1_vanilla",
                step=step,
                total_steps=total_steps,
                episodes_done=episodes_done,
                ep_return=ep_return,
                ep_len=ep_len,
                buf_size=len(buf),
                epsilon=eps,
                lambda_cost=float(env.lambda_cost),
                mean_consistency=mean_cons,
            )
            t_ep_stop = float(info.get("episode_mean_iou_teacher", float("nan")))
            if ref_iou_early_stop is not None and ref_iou_early_stop.on_episode_end(step, t_ep_stop):
                break
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0
            ep_actions.clear()

        if step >= ls and step % train_freq == 0 and len(buf) >= batch_size:
            for _ in range(gu):
                b_o, b_a, b_r, b_no, b_d = buf.sample(batch_size)
                b_o = augment_obs_batch(b_o, obs_noise_std, rng)
                b_no = augment_obs_batch(b_no, obs_noise_std, rng)
                o = torch.from_numpy(b_o).float().to(dev)
                a_ = torch.from_numpy(b_a).long().to(dev)
                r_ = torch.from_numpy(b_r).float().to(dev)
                no = torch.from_numpy(b_no).float().to(dev)
                d_ = torch.from_numpy(b_d).float().to(dev)
                q_sa = q(o).gather(1, a_.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_online = q(no)
                    next_act = next_online.argmax(dim=1, keepdim=True)
                    q_next = target(no).gather(1, next_act).squeeze(1)
                    target_q = r_ + gamma * (1.0 - d_) * q_next
                loss = nn.functional.smooth_l1_loss(q_sa, target_q, beta=huber_beta)
                opt.zero_grad()
                loss.backward()
                gn = float(nn.utils.clip_grad_norm_(q.parameters(), 10.0))
                opt.step()
                if sched is not None:
                    sched.step()
                losses_since_log.append(float(loss.detach().cpu()))
                td_abs_since_log.append(float((q_sa.detach() - target_q).abs().mean().cpu()))
                grad_norms_since_log.append(gn)
                if use_soft_target:
                    polyak_update(target, q, target_tau)
            if not use_soft_target and step % target_update_every == 0:
                target.load_state_dict(q.state_dict())

        if int(log_every) > 0 and step % int(log_every) == 0:
            log_metrics()

    pbar.close()
    if int(log_every) > 0 and step > 0 and step % int(log_every) != 0:
        log_metrics()
    if csv_file is not None:
        csv_file.close()

    q.eval()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q_state_dict": q.state_dict(),
            "target_state_dict": target.state_dict(),
            "step": step,
            "lambda_cost_final": env.lambda_cost,
            "state_dim": state_dim,
            "architecture": "vanilla",
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "fixed_lambda": fixed_lambda,
            "group": "g1_vanilla",
            "obs_stack_k": int(stack_k),
            "ablation": ablation,
        },
        save_path,
    )
    print_training_done("g1_vanilla", save_path=save_path, final_lambda=float(env.lambda_cost))
