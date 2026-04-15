"""gym env: fish video + gated YOLO + flow reuse. reward uses baseline boxes when file given."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utilities.actions import FULL_DETECT, LIGHT_UPDATE, REUSE
from utilities.baseline_npz import load_baseline_npz
from utilities.detector import load_yolo, predict_best_box
from utilities.geometry import clip_xyxy, iou_xyxy, xyxy_to_cxcywh_norm
from utilities.state import STATE_DIM


def _mean_abs_diff_norm(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(g1.astype(np.float32) - g0.astype(np.float32))) / 255.0)


def _flow_shift_box(
    prev_bgr: np.ndarray, curr_bgr: np.ndarray, box_xyxy: np.ndarray, w: int, h: int
) -> np.ndarray:
    prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return box_xyxy.copy()
    xs = np.linspace(x1, x2, num=5)
    ys = np.linspace(y1, y2, num=5)
    pts = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
    pts = pts.reshape(-1, 1, 2)
    next_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, pts, None)
    if next_pts is None or st is None:
        return box_xyxy.copy()
    st = st.ravel()
    good = st == 1
    if int(good.sum()) < 4:
        return box_xyxy.copy()
    d = (next_pts[good] - pts[good]).reshape(-1, 2)
    dx = float(np.median(d[:, 0]))
    dy = float(np.median(d[:, 1]))
    shifted = box_xyxy.astype(np.float32) + np.array([dx, dy, dx, dy], dtype=np.float32)
    return clip_xyxy(shifted, w, h)


def _default_box_xyxy(w: int, h: int) -> np.ndarray:
    return clip_xyxy(
        np.array([0.25 * w, 0.25 * h, 0.75 * w, 0.75 * h], dtype=np.float32),
        w,
        h,
    )


class FishTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        video_path: Path,
        yolo_weights: Path,
        *,
        baseline_npz: Path | None = None,
        lambda_cost: float = 0.4,
        max_episode_steps: int = 200,
        imgsz: int = 640,
        device: str = "mps",
        conf: float = 0.25,
        action_costs: tuple[float, float, float] = (1.0, 0.25, 0.0),
        random_start: bool = True,
        seed: int | None = None,
        ssf_reward_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self._np_random: np.random.Generator | None = None
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.video_path = Path(video_path)
        self.yolo_weights = Path(yolo_weights)
        self.baseline_npz: Path | None = Path(baseline_npz) if baseline_npz else None
        self.lambda_cost = float(lambda_cost)
        self.max_episode_steps = int(max_episode_steps)
        self.imgsz = int(imgsz)
        self.device = device
        self.conf = float(conf)
        self.action_costs = action_costs
        self.random_start = random_start
        self.ssf_reward_penalty = float(ssf_reward_penalty)
        self._ssf_scale = float(max(50, max_episode_steps // 2))

        self._cap_w = 0
        self._cap_h = 0
        self._frames: list[np.ndarray] = []
        self._baseline: dict[str, Any] | None = None
        self._has_baseline = False
        self._model = None

        self._load_video()
        n_vid = len(self._frames)
        if self.baseline_npz is not None and self.baseline_npz.is_file():
            self._baseline = load_baseline_npz(self.baseline_npz)
            n = int(self._baseline["n_frames"])
            if n_vid != n:
                raise ValueError(
                    f"video frames {n_vid} != baseline npz {n} — fix paths or regenerate npz"
                )
            self._has_baseline = True

        self._min_start = 1
        if self._has_baseline and self._baseline is not None:
            n = int(self._baseline["n_frames"])
            for t in range(1, n):
                bx = self._baseline["boxes"][t]
                if bx[2] > bx[0] and bx[3] > bx[1]:
                    self._min_start = int(t)
                    break

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

        self._t: int = 0
        self._ep_len: int = 0
        self._start_idx: int = 0
        self._pred_xyxy = np.zeros(4, dtype=np.float32)
        self._conf: float = 0.0
        self._prev_cxcy: np.ndarray = np.zeros(2, dtype=np.float32)
        self._episode_compute: float = 0.0
        self._episode_consistency_sum: float = 0.0
        self._episode_baseline_iou_sum: float = 0.0
        self._steps_since_full: int = 0
        self._pred_before_last: np.ndarray = np.zeros(4, dtype=np.float32)

    def seed(self, seed: int | None = None) -> None:
        self._np_random = np.random.default_rng(seed)

    def _load_video(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {self.video_path}")
        self._cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            self._frames.append(fr)
        cap.release()
        if not self._frames:
            raise RuntimeError("video has zero frames")
        print(
            f"[FishTrackingEnv] loaded {len(self._frames)} frames {self._cap_w}x{self._cap_h}",
            flush=True,
        )

    def _ensure_model(self) -> None:
        if self._model is None:
            print(
                f"[FishTrackingEnv] loading YOLO {self.yolo_weights} device={self.device}",
                flush=True,
            )
            self._model = load_yolo(self.yolo_weights, device=self.device)
            print("[FishTrackingEnv] YOLO ready", flush=True)

    def _get_obs(self, prev_idx: int, curr_idx: int) -> np.ndarray:
        w, h = self._cap_w, self._cap_h
        fd = _mean_abs_diff_norm(self._frames[prev_idx], self._frames[curr_idx])
        cxcywh = xyxy_to_cxcywh_norm(self._pred_xyxy, w, h)
        vx = float(cxcywh[0] - self._prev_cxcy[0])
        vy = float(cxcywh[1] - self._prev_cxcy[1])
        flow_ref = _flow_shift_box(
            self._frames[prev_idx],
            self._frames[curr_idx],
            self._pred_before_last,
            w,
            h,
        )
        align_flow = float(iou_xyxy(self._pred_xyxy, flow_ref))
        ssf_n = min(self._steps_since_full / self._ssf_scale, 1.0)
        return np.array(
            [
                fd,
                float(cxcywh[0]),
                float(cxcywh[1]),
                float(cxcywh[2]),
                float(cxcywh[3]),
                float(self._conf),
                vx,
                vy,
                align_flow,
                float(ssf_n),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        n = len(self._frames)
        if n < 3:
            raise RuntimeError("need at least 3 frames")
        if self._has_baseline and self._min_start >= n - 1:
            raise RuntimeError("baseline has no valid box")

        rng = self._np_random or np.random.default_rng()
        low = self._min_start
        high_excl = n - 1
        if self.random_start:
            if low >= high_excl:
                self._start_idx = low
            else:
                self._start_idx = int(rng.integers(low, high_excl))
        else:
            self._start_idx = low

        self._t = self._start_idx
        self._ep_len = 0
        self._episode_compute = 0.0
        self._episode_consistency_sum = 0.0
        self._episode_baseline_iou_sum = 0.0
        self._steps_since_full = 0

        self._ensure_model()
        w, h = self._cap_w, self._cap_h
        b, sc = predict_best_box(
            self._model,
            self._frames[self._t],
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
        )
        if b is not None:
            self._pred_xyxy = clip_xyxy(b, w, h)
            self._conf = float(sc)
        elif self._has_baseline and self._baseline is not None:
            self._pred_xyxy = self._baseline["boxes"][self._t].copy()
            self._conf = 0.0
        else:
            self._pred_xyxy = _default_box_xyxy(w, h)
            self._conf = 0.0

        self._pred_before_last = self._pred_xyxy.copy()
        cxcywh = xyxy_to_cxcywh_norm(self._pred_xyxy, w, h)
        self._prev_cxcy = np.array([cxcywh[0], cxcywh[1]], dtype=np.float32)

        prev_idx = self._t - 1
        obs = self._get_obs(prev_idx, self._t)
        return obs, {
            "frame_idx": self._t,
            "start_idx": self._start_idx,
            "lambda_cost": self.lambda_cost,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        w, h = self._cap_w, self._cap_h
        if self._t >= len(self._frames):
            raise RuntimeError("step after done")

        prev_idx = self._t - 1
        curr_idx = self._t
        cost = self.action_costs[int(action)]
        self._episode_compute += cost

        pred_before = self._pred_xyxy.copy()
        self._ensure_model()
        if action == FULL_DETECT:
            b, sc = predict_best_box(
                self._model,
                self._frames[curr_idx],
                imgsz=self.imgsz,
                device=self.device,
                conf=self.conf,
            )
            if b is not None:
                self._pred_xyxy = clip_xyxy(b, w, h)
                self._conf = float(sc)
            self._steps_since_full = 0
        elif action == LIGHT_UPDATE:
            self._pred_xyxy = _flow_shift_box(
                self._frames[prev_idx],
                self._frames[curr_idx],
                self._pred_xyxy,
                w,
                h,
            )
            self._conf = max(0.0, self._conf * 0.95)
            self._steps_since_full += 1
        else:
            self._steps_since_full += 1

        flow_ref = _flow_shift_box(
            self._frames[prev_idx],
            self._frames[curr_idx],
            pred_before,
            w,
            h,
        )
        consistency = float(iou_xyxy(self._pred_xyxy, flow_ref))
        self._episode_consistency_sum += consistency

        iou_baseline = float("nan")
        if self._has_baseline and self._baseline is not None:
            ref = self._baseline["boxes"][curr_idx]
            iou_baseline = float(iou_xyxy(self._pred_xyxy, ref))
            self._episode_baseline_iou_sum += iou_baseline

        if self._has_baseline:
            reward = float(iou_baseline) - self.lambda_cost * cost
        else:
            reward = consistency - self.lambda_cost * cost

        if self.ssf_reward_penalty > 0.0:
            ssf_n = min(self._steps_since_full / self._ssf_scale, 1.0)
            reward -= self.ssf_reward_penalty * float(ssf_n)

        cxcywh = xyxy_to_cxcywh_norm(self._pred_xyxy, w, h)
        self._prev_cxcy = np.array([cxcywh[0], cxcywh[1]], dtype=np.float32)

        self._pred_before_last = pred_before.copy()

        self._ep_len += 1
        self._t += 1
        terminated = False
        truncated = self._ep_len >= self.max_episode_steps or self._t >= len(self._frames)

        if not truncated:
            next_obs = self._get_obs(self._t - 1, self._t)
        else:
            next_obs = np.zeros(STATE_DIM, dtype=np.float32)

        mean_cost_step = self._episode_compute / max(self._ep_len, 1)
        mean_cons_ep = self._episode_consistency_sum / max(self._ep_len, 1)
        mean_bl_ep = (
            self._episode_baseline_iou_sum / max(self._ep_len, 1)
            if self._has_baseline
            else float("nan")
        )
        # keep old key names so rollout() in utilities/evaluate still works
        info = {
            "iou": consistency,
            "consistency_iou": consistency,
            "iou_teacher": iou_baseline,
            "action_cost": cost,
            "frame_idx": curr_idx,
            "episode_compute": self._episode_compute,
            "episode_mean_consistency": mean_cons_ep,
            "episode_mean_iou": mean_cons_ep,
            "episode_mean_iou_teacher": mean_bl_ep,
            "episode_mean_cost": mean_cost_step,
            "lambda_cost": self.lambda_cost,
            "reward_mode": "baseline_iou" if self._has_baseline else "flow_consistency",
            "steps_since_full": int(self._steps_since_full),
            "ssf_norm": float(min(self._steps_since_full / self._ssf_scale, 1.0)),
        }
        return next_obs, float(reward), terminated, truncated, info
