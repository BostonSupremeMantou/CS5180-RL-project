"""Stage 5：old_research_targets.md §6.4（PER / Soft Q / C51 / GRU）。"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from stage_1.env.fish_tracking_env import STATE_DIM, FishTrackingEnv
from stage_1.models.dqn import TrackingDQN, select_action_epsilon_greedy
from stage_1.training.replay_buffer import ReplayBuffer
from stage_1.training.train_dqn import _augment_obs_batch, _polyak_update
from stage_1.utils.paths import DEFAULT_TEACHER_NPZ, VIDEO_PATH, WEIGHTS_PATH
from stage_2.env.wrappers import build_stage2_env, stage2_state_dim
from stage_2.training.stage2_train_args import apply_all_presets, build_stage2_train_argparser
from stage_5.models.c51_net import C51TrackingDQN, project_c51
from stage_5.models.gru_dueling import GRUDuelingDQN
from stage_5.training.per_buffer import PrioritizedReplayBuffer


def _teacher_ma_value(ep_teacher_ma: deque) -> float:
    if not ep_teacher_ma:
        return float("nan")
    arr = np.asarray(list(ep_teacher_ma), dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def train_stage5(
    *,
    stage5_mode: str,
    video_path: Path | None,
    teacher_npz: Path | None,
    stack_k: int,
    ablation: str,
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
    log_every: int,
    metrics_csv: Path | None,
    metrics_ma_episodes: int,
    fixed_lambda: bool,
    target_mean_cost: float,
    lambda_lr: float,
    lambda_initial: float,
    lambda_min: float,
    lambda_max: float,
    huber_beta: float,
    hidden_dim: int,
    dropout: float,
    obs_noise_std: float,
    policy_obs_noise_std: float,
    grad_updates: int,
    preset_name: str,
    stage2_preset_name: str,
    ssf_reward_penalty: float,
    epsilon_tail: float | None,
    epsilon_tail_steps: int,
    lambda_teacher_iou_floor: float | None,
    lambda_teacher_iou_below: float,
    softq_tau: float = 0.5,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
) -> None:
    mode = stage5_mode.lower().strip()
    if mode not in {"per", "softq", "c51", "gru"}:
        raise ValueError(f"unknown --stage5-mode: {stage5_mode}")

    dev = torch.device("cpu")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    lam = float(lambda_initial if not fixed_lambda else lambda_cost)
    vp = Path(video_path) if video_path is not None else VIDEO_PATH
    base = FishTrackingEnv(
        vp,
        WEIGHTS_PATH,
        teacher_npz=teacher_npz,
        lambda_cost=lam,
        max_episode_steps=max_episode_steps,
        imgsz=imgsz,
        device=device_s,
        random_start=True,
        seed=seed,
        ssf_reward_penalty=ssf_reward_penalty,
    )
    env = build_stage2_env(base, stack_k=stack_k, ablation=ablation)
    state_dim = stage2_state_dim(stack_k)

    if mode == "c51":
        q = C51TrackingDQN(
            state_dim=state_dim, hidden=hidden_dim, dropout=dropout
        ).to(dev)
        target = C51TrackingDQN(
            state_dim=state_dim, hidden=hidden_dim, dropout=dropout
        ).to(dev)
        arch = "s5_c51"
    elif mode == "gru":
        if stack_k < 2:
            raise SystemExit("§6.4.3 GRU 模式需要 stack_k>=2")
        q = GRUDuelingDQN(
            state_dim=state_dim,
            stack_k=stack_k,
            raw_dim=STATE_DIM,
            hidden=hidden_dim,
            gru_hidden=hidden_dim,
            dropout=dropout,
        ).to(dev)
        target = GRUDuelingDQN(
            state_dim=state_dim,
            stack_k=stack_k,
            raw_dim=STATE_DIM,
            hidden=hidden_dim,
            gru_hidden=hidden_dim,
            dropout=dropout,
        ).to(dev)
        arch = "s5_gru"
    else:
        q = TrackingDQN(state_dim=state_dim, hidden=hidden_dim, dropout=dropout).to(dev)
        target = TrackingDQN(
            state_dim=state_dim, hidden=hidden_dim, dropout=dropout
        ).to(dev)
        arch = "dueling_double"

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

    if mode == "per":
        buf: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(
            replay_capacity,
            state_dim=state_dim,
            rng=rng,
            alpha=per_alpha,
        )
    else:
        buf = ReplayBuffer(replay_capacity, state_dim=state_dim, rng=rng)

    def epsilon_at(step_v: int) -> float:
        if step_v < epsilon_decay_steps:
            return float(
                epsilon_end
                + (epsilon_start - epsilon_end)
                * max(0.0, 1.0 - step_v / float(max(1, epsilon_decay_steps)))
            )
        if epsilon_tail is None or int(epsilon_tail_steps) <= 0:
            return float(epsilon_end)
        t = step_v - int(epsilon_decay_steps)
        if t >= int(epsilon_tail_steps):
            return float(epsilon_tail)
        return float(epsilon_end) + (float(epsilon_tail) - float(epsilon_end)) * (
            t / float(max(1, int(epsilon_tail_steps)))
        )

    print(
        f"[Stage5] mode={mode} 首次 reset… stack_k={stack_k} "
        f"log_every={log_every} total_steps={total_steps}",
        flush=True,
    )
    obs, _ = env.reset(seed=seed)
    print(f"[Stage5] 训练开始 obs_dim={obs.shape[0]} arch={arch}", flush=True)

    ep_return = 0.0
    ep_len = 0
    ep_actions: list[int] = []
    step = 0
    pbar = tqdm(total=total_steps, desc=f"Stage5-{mode}")

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

    def select_a(obs_f: np.ndarray, eps: float) -> int:
        if mode == "c51":
            if rng.random() < eps:
                return int(rng.integers(0, 3))
            t = torch.from_numpy(obs_f).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                return int(q.expected_q(t).argmax(dim=1).item())
        return select_action_epsilon_greedy(q, obs_f, eps, 3, dev, rng)

    def train_step_batch() -> None:
        nonlocal losses_since_log, td_abs_since_log, grad_norms_since_log
        beta = min(
            1.0,
            float(per_beta_start)
            + (1.0 - float(per_beta_start)) * (step / float(max(1, total_steps))),
        )
        if mode == "per":
            assert isinstance(buf, PrioritizedReplayBuffer)
            b_o, b_a, b_r, b_no, b_d, idxs, iw = buf.sample(batch_size, beta)
            iw_t = torch.from_numpy(iw).float().to(dev)
        else:
            b_o, b_a, b_r, b_no, b_d = buf.sample(batch_size)
            idxs = None
            iw_t = None
        b_o = _augment_obs_batch(b_o, obs_noise_std, rng)
        b_no = _augment_obs_batch(b_no, obs_noise_std, rng)
        o = torch.from_numpy(b_o).float().to(dev)
        a = torch.from_numpy(b_a).long().to(dev)
        r = torch.from_numpy(b_r).float().to(dev)
        no = torch.from_numpy(b_no).float().to(dev)
        d = torch.from_numpy(b_d).float().to(dev)

        if mode == "c51":
            logits = q(o)
            logits_sa = logits.gather(1, a.view(-1, 1, 1).expand(-1, 1, q.n_atoms)).squeeze(1)
            with torch.no_grad():
                q_mean_online = q.expected_q(no)
                next_a = q_mean_online.argmax(dim=1)
                next_logits_t = target(no)
                next_probs = F.softmax(next_logits_t, dim=-1)
                next_probs = next_probs[torch.arange(o.size(0), device=dev), next_a]
                m = project_c51(next_probs, r, d, q.support, gamma)
            loss_vec = -(m * F.log_softmax(logits_sa, dim=-1)).sum(dim=-1)
            if iw_t is not None:
                loss = (loss_vec * iw_t).mean()
            else:
                loss = loss_vec.mean()
            with torch.no_grad():
                td_abs = (logits_sa - (m.clamp(min=1e-8)).log()).abs().mean()
        elif mode == "softq":
            q_sa = q(o).gather(1, a.unsqueeze(1)).squeeze(1)
            tau = float(softq_tau)
            with torch.no_grad():
                next_logits = target(no)
                v_soft = tau * torch.logsumexp(next_logits / tau, dim=1)
                target_q = r + gamma * (1.0 - d) * v_soft
            loss_vec = F.smooth_l1_loss(q_sa, target_q, beta=huber_beta, reduction="none")
            if iw_t is not None:
                loss = (loss_vec * iw_t).mean()
            else:
                loss = loss_vec.mean()
            td_abs = (q_sa.detach() - target_q).abs().mean()
        else:
            q_sa = q(o).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_online = q(no)
                next_act = next_online.argmax(dim=1, keepdim=True)
                q_next = target(no).gather(1, next_act).squeeze(1)
                target_q = r + gamma * (1.0 - d) * q_next
            loss_vec = F.smooth_l1_loss(q_sa, target_q, beta=huber_beta, reduction="none")
            if iw_t is not None:
                loss = (loss_vec * iw_t).mean()
            else:
                loss = loss_vec.mean()
            td_abs = (q_sa.detach() - target_q).abs().mean()

        opt.zero_grad()
        loss.backward()
        gn = float(nn.utils.clip_grad_norm_(q.parameters(), 10.0))
        opt.step()
        if sched is not None:
            sched.step()
        losses_since_log.append(float(loss.detach().cpu()))
        td_abs_since_log.append(float(td_abs.detach().cpu()))
        grad_norms_since_log.append(gn)
        if isinstance(buf, PrioritizedReplayBuffer) and idxs is not None:
            td_err = (q_sa.detach() - target_q).abs().cpu().numpy()
            buf.update_priorities(idxs, td_err)

    def log_metrics() -> None:
        nonlocal losses_since_log, td_abs_since_log, grad_norms_since_log
        buf_n = len(buf)
        loss_m = float(np.mean(losses_since_log)) if losses_since_log else float("nan")
        td_m = float(np.mean(td_abs_since_log)) if td_abs_since_log else float("nan")
        gn_m = float(np.mean(grad_norms_since_log)) if grad_norms_since_log else float("nan")

        q_pm = q_tm = float("nan")
        if buf_n >= batch_size:
            if isinstance(buf, PrioritizedReplayBuffer):
                b_o, b_a, b_r, b_no, b_d, _, _ = buf.sample(batch_size, 1.0)
            else:
                b_o, b_a, b_r, b_no, b_d = buf.sample(batch_size)
            b_o = _augment_obs_batch(b_o, obs_noise_std, rng)
            b_no = _augment_obs_batch(b_no, obs_noise_std, rng)
            o = torch.from_numpy(b_o).float().to(dev)
            a = torch.from_numpy(b_a).long().to(dev)
            r = torch.from_numpy(b_r).float().to(dev)
            no = torch.from_numpy(b_no).float().to(dev)
            d = torch.from_numpy(b_d).float().to(dev)
            with torch.no_grad():
                if mode == "c51":
                    logits = q(o)
                    logits_sa = logits.gather(
                        1, a.view(-1, 1, 1).expand(-1, 1, q.n_atoms)
                    ).squeeze(1)
                    q_mean_online = q.expected_q(no)
                    next_a = q_mean_online.argmax(dim=1)
                    next_probs = F.softmax(target(no), dim=-1)[
                        torch.arange(o.size(0), device=dev), next_a
                    ]
                    m = project_c51(next_probs, r, d, q.support, gamma)
                    q_pm = float((F.softmax(logits_sa, -1) * q.support.view(1, -1)).sum(-1).mean().cpu())
                    q_tm = float((m * q.support.view(1, -1)).sum(-1).mean().cpu())
                else:
                    q_all = q(o)
                    q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
                    if mode == "softq":
                        tau = float(softq_tau)
                        next_logits = target(no)
                        v_soft = tau * torch.logsumexp(next_logits / tau, dim=1)
                        tq = r + gamma * (1.0 - d) * v_soft
                    else:
                        next_online = q(no)
                        next_act = next_online.argmax(dim=1, keepdim=True)
                        q_next = target(no).gather(1, next_act).squeeze(1)
                        tq = r + gamma * (1.0 - d) * q_next
                    q_pm = float(q_sa.mean().cpu())
                    q_tm = float(tq.mean().cpu())

        r_ma = float(np.mean(ep_returns_ma)) if ep_returns_ma else float("nan")
        i_ma = float(np.mean(ep_ious_ma)) if ep_ious_ma else float("nan")
        t_ma = _teacher_ma_value(ep_teacher_ma)
        c_ma = float(np.mean(ep_comps_ma)) if ep_comps_ma else float("nan")
        f_ma = float(np.mean(ep_full_frac_ma)) if ep_full_frac_ma else float("nan")
        lam_s = env.lambda_cost
        lr_now = float(opt.param_groups[0]["lr"])

        t_msg = f" tchr={t_ma:.4f}" if not np.isnan(t_ma) else ""
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
                    f"{r_ma:.8f}" if not np.isnan(r_ma) else "",
                    f"{i_ma:.8f}" if not np.isnan(i_ma) else "",
                    f"{t_ma:.8f}" if not np.isnan(t_ma) else "",
                    f"{c_ma:.8f}" if not np.isnan(c_ma) else "",
                    f"{f_ma:.8f}" if not np.isnan(f_ma) else "",
                    episodes_done,
                ]
            )
            csv_file.flush()

        losses_since_log = []
        td_abs_since_log = []
        grad_norms_since_log = []

    while step < total_steps:
        eps = epsilon_at(step)
        obs_for_policy = obs.astype(np.float32)
        if policy_obs_noise_std > 0:
            obs_for_policy = obs_for_policy + rng.normal(
                0.0, policy_obs_noise_std, size=obs_for_policy.shape
            ).astype(np.float32)
        a = select_a(obs_for_policy, eps)
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
            ep_teacher_ma.append(float(info.get("episode_mean_iou_teacher", float("nan"))))
            ep_comps_ma.append(float(info.get("episode_compute", 0.0)))
            ep_full_frac_ma.append(full_frac)

            if not fixed_lambda:
                mean_cost = float(info.get("episode_mean_cost", 0.0))
                new_lam = lam + lambda_lr * (mean_cost - target_mean_cost)
                lam = float(np.clip(new_lam, lambda_min, lambda_max))
                env.lambda_cost = lam

            t_ep = float(info.get("episode_mean_iou_teacher", float("nan")))
            if (
                lambda_teacher_iou_floor is not None
                and not np.isnan(t_ep)
                and float(t_ep) < float(lambda_teacher_iou_below)
            ):
                lam = max(lam, float(lambda_teacher_iou_floor))
                lam = float(np.clip(lam, lambda_min, lambda_max))
                env.lambda_cost = lam

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
                train_step_batch()
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
    extra: dict = {
        "stage5_mode": mode,
        "softq_tau": float(softq_tau) if mode == "softq" else None,
        "per_alpha": float(per_alpha) if mode == "per" else None,
    }
    if mode == "gru":
        extra["gru_hidden_dim"] = int(hidden_dim)
    torch.save(
        {
            "q_state_dict": q.state_dict(),
            "target_state_dict": target.state_dict(),
            "step": step,
            "lambda_cost_final": env.lambda_cost,
            "state_dim": state_dim,
            "architecture": arch,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "fixed_lambda": fixed_lambda,
            "target_mean_cost": target_mean_cost if not fixed_lambda else None,
            "train_preset": preset_name,
            "stage2_preset": stage2_preset_name,
            "stage": 5,
            "obs_stack_k": int(stack_k),
            "ablation": ablation,
            "obs_noise_std": obs_noise_std,
            "policy_obs_noise_std": policy_obs_noise_std,
            "target_tau": target_tau,
            "ssf_reward_penalty": float(ssf_reward_penalty),
            "n_step": 1,
            "epsilon_tail": float(epsilon_tail) if epsilon_tail is not None else None,
            "epsilon_tail_steps": int(epsilon_tail_steps),
            "lambda_teacher_iou_floor": float(lambda_teacher_iou_floor)
            if lambda_teacher_iou_floor is not None
            else None,
            "lambda_teacher_iou_below": float(lambda_teacher_iou_below),
            **extra,
        },
        save_path,
    )
    print(
        f"[ok] Stage5 已保存 {save_path} mode={mode} λ_end={env.lambda_cost:.4f} "
        f"stack_k={stack_k}",
        flush=True,
    )


def main() -> None:
    p = build_stage2_train_argparser()
    p.add_argument(
        "--stage5-mode",
        choices=("per", "softq", "c51", "gru"),
        required=True,
        help="§6.4：per | softq（离散最大熵备份）| c51 | gru",
    )
    p.add_argument("--softq-tau", type=float, default=0.5, help="Soft Q 的 τ（logsumexp 温度）")
    p.add_argument("--per-alpha", type=float, default=0.6, help="PER 优先级指数 α")
    p.add_argument("--per-beta-start", type=float, default=0.4, help="PER IS 的 β 初值（线性升到 1）")
    args = p.parse_args()
    apply_all_presets(args)
    if getattr(args, "n_step", 1) != 1:
        print("[warn] Stage5 当前固定 1-step TD，忽略 --n-step != 1", flush=True)

    from stage_2.utils.paths import STAGE2_CHECKPOINTS
    from stage_5.utils.paths import STAGE5_CHECKPOINTS

    if args.save == STAGE2_CHECKPOINTS / "dqn_stage2.pt":
        args.save = STAGE5_CHECKPOINTS / f"dqn_stage5_{args.stage5_mode}.pt"

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

    train_stage5(
        stage5_mode=str(args.stage5_mode),
        video_path=vp,
        teacher_npz=tn,
        stack_k=args.stack_k,
        ablation=args.ablation,
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
        stage2_preset_name=str(args.stage2_preset),
        ssf_reward_penalty=float(getattr(args, "ssf_reward_penalty", 0.0)),
        epsilon_tail=getattr(args, "epsilon_tail", None),
        epsilon_tail_steps=int(getattr(args, "epsilon_tail_steps", 0)),
        lambda_teacher_iou_floor=getattr(args, "lambda_teacher_iou_floor", None),
        lambda_teacher_iou_below=float(getattr(args, "lambda_teacher_iou_below", 0.35)),
        softq_tau=float(getattr(args, "softq_tau", 0.5)),
        per_alpha=float(getattr(args, "per_alpha", 0.6)),
        per_beta_start=float(getattr(args, "per_beta_start", 0.4)),
    )


if __name__ == "__main__":
    main()
