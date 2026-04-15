from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from stage_1.env.fish_tracking_env import STATE_DIM, FishTrackingEnv
from stage_1.models.dqn import TrackingDQN, select_action_epsilon_greedy
from stage_1.training.replay_buffer import ReplayBuffer
from stage_1.training.train_args import apply_train_preset, build_train_argparser
from stage_1.utils.paths import DEFAULT_TEACHER_NPZ, VIDEO_PATH, WEIGHTS_PATH


def _augment_obs_batch(obs: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return obs
    return (obs + rng.normal(0.0, std, size=obs.shape)).astype(np.float32)


def _polyak_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), online.parameters(), strict=True):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def train(
    *,
    video_path: Path | None = None,
    teacher_npz: Path | None = None,
    total_steps: int,
    learning_starts: int,
    train_freq: int,
    batch_size: int,
    gamma: float,
    lr: float,
    lr_min: float,
    weight_decay: float,
    use_cosine_lr: bool,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    target_update_every: int,
    target_tau: float,
    replay_capacity: int,
    lambda_cost: float,
    max_episode_steps: int,
    imgsz: int,
    device_s: str,
    seed: int,
    save_path: Path,
    log_every: int = 500,
    metrics_csv: Path | None = None,
    metrics_ma_episodes: int = 20,
    fixed_lambda: bool = False,
    target_mean_cost: float = 0.32,
    lambda_lr: float = 0.03,
    lambda_initial: float = 0.45,
    lambda_min: float = 0.02,
    lambda_max: float = 1.5,
    huber_beta: float = 1.0,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    obs_noise_std: float = 0.0,
    policy_obs_noise_std: float = 0.0,
    grad_updates: int = 1,
    preset_name: str = "none",
) -> None:
    dev = torch.device("cpu")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    lam = float(lambda_initial if not fixed_lambda else lambda_cost)
    vp = Path(video_path) if video_path is not None else VIDEO_PATH
    env = FishTrackingEnv(
        vp,
        WEIGHTS_PATH,
        teacher_npz=teacher_npz,
        lambda_cost=lam,
        max_episode_steps=max_episode_steps,
        imgsz=imgsz,
        device=device_s,
        random_start=True,
        seed=seed,
    )

    q = TrackingDQN(
        state_dim=STATE_DIM, hidden=hidden_dim, dropout=dropout
    ).to(dev)
    target = TrackingDQN(
        state_dim=STATE_DIM, hidden=hidden_dim, dropout=dropout
    ).to(dev)
    target.load_state_dict(q.state_dict())
    target.train()
    q.train()

    opt = torch.optim.Adam(q.parameters(), lr=lr, weight_decay=weight_decay)
    gu = max(1, int(grad_updates))
    est_grad_steps = max(
        1, (total_steps - max(learning_starts, 0)) // max(train_freq, 1) * gu
    )
    sched: torch.optim.lr_scheduler.LRScheduler | None = None
    if use_cosine_lr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=est_grad_steps, eta_min=lr_min
        )

    buf = ReplayBuffer(replay_capacity, state_dim=STATE_DIM, rng=rng)

    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    ep_len = 0
    ep_actions: list[int] = []
    step = 0
    grad_step_count = 0
    pbar = tqdm(total=total_steps, desc="DQN")

    losses_since_log: list[float] = []
    td_abs_since_log: list[float] = []
    grad_norms_since_log: list[float] = []

    ep_returns_ma = deque(maxlen=max(1, metrics_ma_episodes))
    ep_ious_ma = deque(maxlen=max(1, metrics_ma_episodes))
    ep_teacher_ma = deque(maxlen=max(1, metrics_ma_episodes))
    ep_comps_ma = deque(maxlen=max(1, metrics_ma_episodes))
    ep_full_frac_ma = deque(maxlen=max(1, metrics_ma_episodes))

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
                "q_pred_mean",
                "q_target_mean",
                "ep_ret_ma",
                "ep_consistency_ma",
                "ep_teacher_iou_ma",
                "ep_comp_ma",
                "ep_full_frac_ma",
                "episodes_done",
            ]
        )

    episodes_done = 0
    use_soft_target = target_tau > 0.0

    def log_metrics() -> None:
        nonlocal losses_since_log, td_abs_since_log, grad_norms_since_log
        buf_n = len(buf)
        loss_m = float(np.mean(losses_since_log)) if losses_since_log else float("nan")
        td_m = float(np.mean(td_abs_since_log)) if td_abs_since_log else float("nan")
        gn_m = float(np.mean(grad_norms_since_log)) if grad_norms_since_log else float("nan")

        q_pm = q_tm = float("nan")
        if buf_n >= batch_size:
            b_o, b_a, b_r, b_no, b_d = buf.sample(batch_size)
            b_o = _augment_obs_batch(b_o, obs_noise_std, rng)
            b_no = _augment_obs_batch(b_no, obs_noise_std, rng)
            o = torch.from_numpy(b_o).float().to(dev)
            a = torch.from_numpy(b_a).long().to(dev)
            r = torch.from_numpy(b_r).float().to(dev)
            no = torch.from_numpy(b_no).float().to(dev)
            d = torch.from_numpy(b_d).float().to(dev)
            with torch.no_grad():
                q_all = q(o)
                q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
                next_online = q(no)
                next_act = next_online.argmax(dim=1, keepdim=True)
                q_next = target(no).gather(1, next_act).squeeze(1)
                tq = r + gamma * (1.0 - d) * q_next
                q_pm = float(q_sa.mean().cpu())
                q_tm = float(tq.mean().cpu())

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

        t_msg = f" tchr={t_ma:.4f}" if ep_teacher_ma and not np.isnan(t_ma) else ""
        msg = (
            f"[metrics] step={step} ε={eps:.4f} λ={lam_s:.4f} lr={lr_now:.2e} buf={buf_n} | "
            f"loss={loss_m:.5f} td_abs={td_m:.5f} |grad|={gn_m:.4f} | "
            f"Q(s,a)={q_pm:.4f} Q_tgt={q_tm:.4f} | "
            f"ep_ma{metrics_ma_episodes}: ret={r_ma:.2f} cons={i_ma:.4f}{t_msg} "
            f"comp={c_ma:.2f} full%={f_ma:.3f} | episodes={episodes_done}"
        )
        tqdm.write(msg)

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
                    f"{q_pm:.8f}" if not np.isnan(q_pm) else "",
                    f"{q_tm:.8f}" if not np.isnan(q_tm) else "",
                    f"{r_ma:.6f}" if not np.isnan(r_ma) else "",
                    f"{i_ma:.8f}" if not np.isnan(i_ma) else "",
                    f"{t_ma:.8f}" if ep_teacher_ma and not np.isnan(t_ma) else "",
                    f"{c_ma:.6f}" if not np.isnan(c_ma) else "",
                    f"{f_ma:.6f}" if not np.isnan(f_ma) else "",
                    episodes_done,
                ]
            )
            csv_file.flush()

        losses_since_log = []
        td_abs_since_log = []
        grad_norms_since_log = []

    while step < total_steps:
        eps = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1.0 - step / float(max(1, epsilon_decay_steps))
        )
        obs_for_policy = obs.astype(np.float32)
        if policy_obs_noise_std > 0:
            obs_for_policy = obs_for_policy + rng.normal(
                0.0, policy_obs_noise_std, size=obs_for_policy.shape
            ).astype(np.float32)
        a = select_action_epsilon_greedy(q, obs_for_policy, eps, 3, dev, rng)
        next_obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        buf.push(obs, a, r, next_obs, done)
        ep_return += r
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
            t_ep = float(info.get("episode_mean_iou_teacher", float("nan")))
            ep_teacher_ma.append(t_ep)
            ep_comps_ma.append(float(info.get("episode_compute", 0.0)))
            ep_full_frac_ma.append(full_frac)

            if not fixed_lambda:
                mean_cost = float(info.get("episode_mean_cost", 0.0))
                new_lam = lam + lambda_lr * (mean_cost - target_mean_cost)
                lam = float(np.clip(new_lam, lambda_min, lambda_max))
                env.lambda_cost = lam

            t_ep = float(info.get("episode_mean_iou_teacher", float("nan")))
            t_post = {}
            if not np.isnan(t_ep):
                t_post["tchr"] = f"{t_ep:.3f}"
            pbar.set_postfix(
                ret=f"{ep_return:.2f}",
                len=ep_len,
                cons=f"{info.get('episode_mean_consistency', info.get('episode_mean_iou', 0)):.3f}",
                comp=f"{info.get('episode_compute', 0):.1f}",
                full=f"{100*full_frac:.0f}%",
                eps=f"{eps:.2f}",
                lam=f"{env.lambda_cost:.2f}",
                **t_post,
            )
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0
            ep_actions = []

        if step >= learning_starts and step % train_freq == 0 and len(buf) >= batch_size:
            for _ in range(gu):
                b_o, b_a, b_r, b_no, b_d = buf.sample(batch_size)
                b_o = _augment_obs_batch(b_o, obs_noise_std, rng)
                b_no = _augment_obs_batch(b_no, obs_noise_std, rng)
                o = torch.from_numpy(b_o).float().to(dev)
                a = torch.from_numpy(b_a).long().to(dev)
                r = torch.from_numpy(b_r).float().to(dev)
                no = torch.from_numpy(b_no).float().to(dev)
                d = torch.from_numpy(b_d).float().to(dev)
                q_sa = q(o).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_online = q(no)
                    next_act = next_online.argmax(dim=1, keepdim=True)
                    q_next = target(no).gather(1, next_act).squeeze(1)
                    target_q = r + gamma * (1.0 - d) * q_next
                loss = nn.functional.smooth_l1_loss(q_sa, target_q, beta=huber_beta)
                opt.zero_grad()
                loss.backward()
                gn = float(nn.utils.clip_grad_norm_(q.parameters(), 10.0))
                opt.step()
                grad_step_count += 1
                if sched is not None:
                    sched.step()
                losses_since_log.append(float(loss.detach().cpu()))
                td_abs_since_log.append(
                    float((q_sa.detach() - target_q).abs().mean().cpu())
                )
                grad_norms_since_log.append(gn)
                if use_soft_target:
                    _polyak_update(target, q, target_tau)

            if not use_soft_target and step % target_update_every == 0:
                target.load_state_dict(q.state_dict())

        if log_every > 0 and step % log_every == 0:
            log_metrics()

    pbar.close()
    if log_every > 0 and step > 0 and step % log_every != 0:
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
            "state_dim": STATE_DIM,
            "architecture": "dueling_double",
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "fixed_lambda": fixed_lambda,
            "target_mean_cost": target_mean_cost if not fixed_lambda else None,
            "train_preset": preset_name,
            "obs_noise_std": obs_noise_std,
            "policy_obs_noise_std": policy_obs_noise_std,
            "target_tau": target_tau,
        },
        save_path,
    )
    print(f"[ok] 已保存 {save_path} (λ_end={env.lambda_cost:.4f}, preset={preset_name})")


def main() -> None:
    p = build_train_argparser()
    args = p.parse_args()
    apply_train_preset(args)
    if args.no_cosine_lr:
        args.use_cosine_lr = False
    elif args.cosine_lr:
        args.use_cosine_lr = True
    elif not hasattr(args, "use_cosine_lr"):
        args.use_cosine_lr = False

    vp = Path(args.video_path) if args.video_path is not None else VIDEO_PATH
    if not vp.is_file():
        raise SystemExit(f"找不到训练视频: {vp}")
    tn = None
    if getattr(args, "no_teacher_reward", False):
        print("[info] --no-teacher-reward：自监督光流奖励，不加载 teacher")
    elif args.teacher_npz is not None:
        tnp = Path(args.teacher_npz)
        if tnp.is_file():
            tn = tnp
        else:
            print(f"[warn] --teacher-npz 不存在 ({tnp})，按无 teacher 训练")
    elif DEFAULT_TEACHER_NPZ.is_file():
        tn = DEFAULT_TEACHER_NPZ
        print(f"[info] 使用默认 teacher 参与奖励: {tn}")

    train(
        video_path=vp,
        teacher_npz=tn,
        total_steps=args.total_steps,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        use_cosine_lr=args.use_cosine_lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_every=args.target_update_every,
        target_tau=args.target_tau,
        replay_capacity=args.replay_capacity,
        lambda_cost=args.lambda_cost,
        max_episode_steps=args.max_episode_steps,
        imgsz=args.imgsz,
        device_s=args.device,
        seed=args.seed,
        save_path=args.save,
        log_every=args.log_every,
        metrics_csv=args.metrics_csv,
        metrics_ma_episodes=args.metrics_ma_episodes,
        fixed_lambda=args.fixed_lambda,
        target_mean_cost=args.target_mean_cost,
        lambda_lr=args.lambda_lr,
        lambda_initial=args.lambda_initial,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        huber_beta=args.huber_beta,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        obs_noise_std=args.obs_noise_std,
        policy_obs_noise_std=args.policy_obs_noise_std,
        grad_updates=args.grad_updates,
        preset_name=str(args.preset),
    )


if __name__ == "__main__":
    main()
