from __future__ import annotations

"""
CNN DQN agent for 2048 with:
 - One-hot (4x4x16) input
 - Multi-scale directional CNN (as in the diagram: keep 6 maps)
 - Dueling head (Value + Advantage)
 - Double DQN with illegal-action masking
 - N-step returns (n=3 by default)
 - Symmetry augmentation by random rotations (action remapping)
 - PER-lite sampling (bias toward positive-reward transitions)
 - Reward scaling/optional clipping; Huber loss; grad clipping; target sync

Assumes Game2048 API:
  env = Game2048()
  env.board -> np.ndarray shape (4,4) with integers (0 or powers of two)
  env.score -> int
  env.get_available_actions() -> Sequence[int] for actions {0:Up,1:Down,2:Left,3:Right}
  env.step(action) -> (board, score, done)
"""

import argparse
import csv
import math
import os
import pickle
import random
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from game2048 import Game2048

# --------------------------- Encoding & Augmentation ---------------------------

N_CHANNELS = 16  # empty + 2..2^15 (clamped)


def board_to_onehot(board: np.ndarray, n_channels: int = N_CHANNELS) -> np.ndarray:
    """Return one-hot (4,4,16). ch 0=empty, ch k=2^k (clamped to 15)."""
    onehot = np.zeros((4, 4, n_channels), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            v = int(board[r, c])
            k = 0 if v == 0 else min(int(np.log2(v)), n_channels - 1)
            onehot[r, c, k] = 1.0
    return onehot

# Action map to index according to game2048.py
A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3

# Rotation by k*90 degrees clockwise: action remap tables
ROT_ACTION = {
    0: {A_UP: A_UP,    A_RIGHT: A_RIGHT, A_DOWN: A_DOWN,  A_LEFT: A_LEFT},   # 0°
    1: {A_UP: A_RIGHT, A_RIGHT: A_DOWN,  A_DOWN: A_LEFT,  A_LEFT: A_UP},     # 90° cw
    2: {A_UP: A_DOWN,  A_RIGHT: A_LEFT,  A_DOWN: A_UP,    A_LEFT: A_RIGHT},  # 180°
    3: {A_UP: A_LEFT,  A_RIGHT: A_UP,    A_DOWN: A_RIGHT, A_LEFT: A_DOWN},   # 270° cw
}

def rotate_state(x: np.ndarray, k: int) -> np.ndarray:
    """Rotate one-hot tensor by k*90 degrees clockwise (spatial axes only)."""
    if k % 4 == 0:
        return x
    # np.rot90 uses ccw; use 4-k to get cw
    return np.rot90(x, k=4 - (k % 4), axes=(0, 1)).copy()


def remap_action(a: int, k: int) -> int:
    return ROT_ACTION[k % 4][int(a)]


def remap_actions(actions: Sequence[int], k: int) -> List[int]:
    return [remap_action(a, k) for a in actions]

# ----------------------------- Replay & N-step -------------------------------

@dataclass
class EpisodeStats:
    reward: float
    score: int
    moves: int
    max_tile: int


class ReplayBuffer:
    """
    Uniform buffer with PER-lite sampling toward positive rewards.
    pos_frac: fraction of transitions with positive rewards.
    """
    def __init__(self, capacity: int, pos_frac: float = 0.5) -> None:
        self.capacity = capacity
        self.buf = deque(maxlen=capacity)
        self.pos_frac = float(np.clip(pos_frac, 0.0, 1.0))

    def push(self, s, a, r, s2, done, next_avail) -> None:
        self.buf.append((s, a, r, s2, done, tuple(next_avail)))

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int):
        if self.pos_frac <= 1e-6:
            batch = random.sample(self.buf, batch_size)
        else:
            pos_idx = [i for i, (_, _, r, *_rest) in enumerate(self.buf) if r > 0]
            k_pos = min(int(math.ceil(batch_size * self.pos_frac)), len(pos_idx))
            pos_samples = random.sample(pos_idx, k_pos) if k_pos > 0 else []
            remain = batch_size - len(pos_samples)
            pool = [i for i in range(len(self.buf)) if i not in pos_samples]
            others = random.sample(pool, remain)
            idxs = pos_samples + others
            random.shuffle(idxs)
            batch = [self.buf[i] for i in idxs]
        s, a, r, s2, d, next_avail = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int32),
            np.array(r, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(d, dtype=np.bool_),
            next_avail,
        )

    def state_dict(self):
        return {"capacity": self.capacity, "data": list(self.buf), "pos_frac": self.pos_frac}

    def load_state(self, state):
        if not state:
            return
        self.capacity = int(state.get("capacity", self.capacity))
        self.pos_frac = float(state.get("pos_frac", self.pos_frac))
        data = state.get("data", [])
        self.buf = deque(data, maxlen=self.capacity)


class NStepBuilder:
    """Collect 1-step transitions inside an episode, emit n-step transitions at end."""
    def __init__(self, n: int, gamma: float) -> None:
        assert n >= 1
        self.n = n
        self.gamma = gamma
        self._steps = []  # list of dicts per step

    def add(self, s, a, r, s2, done, next_avail):
        self._steps.append({"s": s, "a": a, "r": r, "s2": s2, "done": done, "next_avail": tuple(next_avail)})

    def emit(self):
        n = self.n
        g = self.gamma
        out = []
        T = len(self._steps)
        for t in range(T):
            R = 0.0
            done_n = False
            last_idx = t
            for i in range(n):
                idx = t + i
                if idx >= T:
                    break
                step = self._steps[idx]
                R += (g ** i) * float(step["r"])
                last_idx = idx
                if step["done"]:
                    done_n = True
                    break
            s0 = self._steps[t]["s"]
            a0 = self._steps[t]["a"]
            sN = self._steps[last_idx]["s2"]
            next_avN = self._steps[last_idx]["next_avail"]
            out.append((s0, a0, R, sN, done_n, next_avN))
        self._steps.clear()
        return out

# ------------------------------ CNN Q-network -------------------------------

def conv_block_h(inp, channels: int):
    # Horizontal 1x2 then 1x2
    h1 = layers.Conv2D(channels, (1, 2), padding="valid", activation="relu")(inp)  # 4x3
    h2 = layers.Conv2D(channels, (1, 2), padding="valid", activation="relu")(h1)   # 4x2
    return h1, h2


def conv_block_v(inp, channels: int):
    # Vertical 2x1 then 2x1
    v1 = layers.Conv2D(channels, (2, 1), padding="valid", activation="relu")(inp)  # 3x4
    v2 = layers.Conv2D(channels, (2, 1), padding="valid", activation="relu")(v1)   # 2x4
    return v1, v2


def cross_maps(h1, v1, channels: int):
    # From first maps, build 3x3 context using the opposite kernels
    x1 = layers.Conv2D(channels, (2, 1), padding="valid", activation="relu")(h1)   # 3x3
    x2 = layers.Conv2D(channels, (1, 2), padding="valid", activation="relu")(v1)   # 3x3
    return x1, x2


def build_qnet_conv_dueling(channels: int = 128, n_actions: int = 4) -> keras.Model:
    inp = keras.Input(shape=(4, 4, N_CHANNELS))
    h1, h2 = conv_block_h(inp, channels)
    v1, v2 = conv_block_v(inp, channels)
    x1, x2 = cross_maps(h1, v1, channels)

    feats = layers.Concatenate()( [
        layers.Flatten()(h1), layers.Flatten()(h2),
        layers.Flatten()(v1), layers.Flatten()(v2),
        layers.Flatten()(x1), layers.Flatten()(x2),
    ])  #  (4*3 + 4*2 + 3*4 + 2*4 + 3*3 + 3*3)*C = 58*C

    trunk = layers.Dense(256, activation="relu")(feats)

    # Dueling head
    adv = layers.Dense(128, activation="relu")(trunk)
    adv = layers.Dense(n_actions, activation=None)(adv)

    val = layers.Dense(128, activation="relu")(trunk)
    val = layers.Dense(1, activation=None)(val)

    # Combine: Q = V + (A - mean(A))
    adv_centered = layers.Lambda(lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True))(adv)
    out = layers.Add()([val, adv_centered])

    return keras.Model(inp, out)

# ------------------------------- DQN Agent ----------------------------------

class DQNAgent:
    def __init__(
        self,
        n_actions: int = 4,
        channels: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-4,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.998,
        target_sync: int = 1000,
        clip_norm: float = 1.0,
        device: str = "/CPU:0",
        seed: int = 42,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_sync = target_sync
        self.clip_norm = clip_norm
        self.step_count = 0

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        with tf.device(device):
            self.q = build_qnet_conv_dueling(channels=channels, n_actions=n_actions)
            self.tgt = build_qnet_conv_dueling(channels=channels, n_actions=n_actions)
            self.tgt.set_weights(self.q.get_weights())
            self.opt = keras.optimizers.Adam(learning_rate=lr)
            self.loss_fn = keras.losses.Huber()

    def select_action(self, state_oh: np.ndarray, available_actions: Sequence[int], explore=True) -> int:
        if not available_actions:
            return 0
        if explore and random.random() < self.eps:
            return random.choice(list(available_actions))
        q = self.q(np.expand_dims(state_oh, 0), training=False).numpy()[0]
        masked = np.full_like(q, -np.inf, dtype=np.float32)
        for a in available_actions:
            masked[a] = q[a]
        best = float(np.max(masked))
        best_actions = [a for a in available_actions if masked[a] >= best - 1e-9]
        return random.choice(best_actions)

    @tf.function
    def _train_step(self, s, a, r, s2, d, next_mask, gamma_n):
        with tf.GradientTape() as tape:
            q_all = self.q(s, training=True)                # (B,4)
            q_sa = tf.gather(q_all, a, batch_dims=1)        # (B,)

            q_next_online = self.q(s2, training=False)
            neg_inf = tf.constant(-1e9, dtype=q_next_online.dtype)
            masked_next = q_next_online + (1.0 - next_mask) * neg_inf
            next_act = tf.argmax(masked_next, axis=1, output_type=tf.int32)

            q_next_target = self.tgt(s2, training=False)
            q_next_target_sa = tf.gather(q_next_target, next_act, batch_dims=1)

            not_done = tf.cast(tf.logical_not(d), tf.float32)
            target = r + not_done * gamma_n * q_next_target_sa
            loss = self.loss_fn(target, q_sa)

        vars = self.q.trainable_variables
        grads = tape.gradient(loss, vars)
        if self.clip_norm and self.clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        pairs = [(g, v) for g, v in zip(grads, vars) if g is not None]
        if pairs:
            self.opt.apply_gradients(pairs)
        return loss

    def update(self, batch, aug_rot_prob: float = 0.5, gamma_n: float | None = None):
        s, a, r, s2, d, next_avail = batch
        B = s.shape[0]

        # Optional symmetry augmentation via random rotations
        if aug_rot_prob > 0:
            ks = np.random.choice([0, 1, 2, 3], size=B, p=[1 - aug_rot_prob, aug_rot_prob/3, aug_rot_prob/3, aug_rot_prob/3])
            s_aug = np.stack([rotate_state(s[i], int(ks[i])) for i in range(B)])
            s2_aug = np.stack([rotate_state(s2[i], int(ks[i])) for i in range(B)])
            a_aug = np.array([remap_action(a[i], int(ks[i])) for i in range(B)], dtype=np.int32)
            next_mask = np.zeros((B, self.n_actions), dtype=np.float32)
            for i in range(B):
                avail = remap_actions(next_avail[i], int(ks[i]))
                if avail:
                    next_mask[i, list(avail)] = 1.0
        else:
            s_aug, s2_aug, a_aug = s, s2, a
            next_mask = np.zeros((B, self.n_actions), dtype=np.float32)
            for i, avail in enumerate(next_avail):
                if avail:
                    next_mask[i, list(avail)] = 1.0

        gamma_eff = self.gamma if gamma_n is None else gamma_n
        loss = self._train_step(
            tf.convert_to_tensor(s_aug, dtype=tf.float32),
            tf.convert_to_tensor(a_aug, dtype=tf.int32),
            tf.convert_to_tensor(r, dtype=tf.float32),
            tf.convert_to_tensor(s2_aug, dtype=tf.float32),
            tf.convert_to_tensor(d, dtype=tf.bool),
            tf.convert_to_tensor(next_mask, dtype=tf.float32),
            tf.convert_to_tensor(gamma_eff, dtype=tf.float32),
        )

        self.step_count += 1
        if self.step_count % self.target_sync == 0:
            self.tgt.set_weights(self.q.get_weights())
        return float(tf.keras.backend.get_value(loss))

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def state_dict(self):
        return {
            "q_weights": self.q.get_weights(),
            "tgt_weights": self.tgt.get_weights(),
            "eps": self.eps,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state) -> None:
        if not state:
            return
        qw = state.get("q_weights")
        if qw is not None:
            self.q.set_weights(qw)
        tw = state.get("tgt_weights")
        if tw is not None:
            self.tgt.set_weights(tw)
        elif qw is not None:
            self.tgt.set_weights(qw)
        self.eps = float(state.get("eps", self.eps))
        self.step_count = int(state.get("step_count", self.step_count))


# --------------------------- Episode & Training Loop -------------------------

def run_episode(
    agent: DQNAgent,
    env: Game2048,
    *,
    train: bool,
    step_penalty: float,
    buffer: ReplayBuffer | None,
    reward_scale: float,
    reward_clip: float | None,
    nstep: int,
    gamma: float,
):
    env.reset()

    total_reward = 0.0
    moves = 0
    max_tile = int(env.board.max())

    # Collect 1-step transitions; convert to n-step at end
    nstep_builder = NStepBuilder(n=max(1, nstep), gamma=gamma)

    s_oh = board_to_onehot(env.board)

    while True:
        avail = env.get_available_actions()
        if not avail:
            break
        a = agent.select_action(s_oh, avail, explore=train)

        prev_score = env.score
        board, score, done = env.step(a)

        raw = float(score - prev_score)
        r = reward_scale * raw + step_penalty
        if reward_clip and reward_clip > 0:
            r = float(np.clip(r, -reward_clip, reward_clip))

        total_reward += r
        moves += 1
        max_tile = max(max_tile, int(board.max()))

        s2_oh = board_to_onehot(board)
        next_avail = env.get_available_actions()

        # Store 1-step; n-step built at end
        nstep_builder.add(s_oh, a, r, s2_oh, done, next_avail)

        s_oh = s2_oh
        if done:
            break

    if buffer is not None and train:
        for tr in nstep_builder.emit():
            buffer.push(*tr)

    if train:
        agent.decay_epsilon()

    return EpisodeStats(reward=total_reward, score=int(env.score), moves=moves, max_tile=max_tile)

def evaluate_agent(
    agent: DQNAgent,
    env: Game2048,
    episodes: int,
    *,
    reward_scale: float,
    reward_clip: float | None,
    eval_writer=None,
    eval_file=None,
) -> EpisodeStats:
    scores: List[int] = []
    rewards: List[float] = []
    moves: List[int] = []
    max_tiles: List[int] = []

    eps0 = agent.eps
    agent.eps = 0.0
    try:
        for epi in range(1, episodes + 1):
            stats = run_episode(
                agent,
                env,
                train=False,
                step_penalty=0.0,
                buffer=None,
                reward_scale=reward_scale,
                reward_clip=reward_clip,
                nstep=1,              # pure 1-step for eval
                gamma=agent.gamma,
            )
            scores.append(stats.score)
            rewards.append(stats.reward)
            moves.append(stats.moves)
            max_tiles.append(stats.max_tile)

            if eval_writer is not None:
                eval_writer.writerow([epi, stats.score, stats.reward, stats.moves, stats.max_tile, 0.0])
                if eval_file is not None:
                    eval_file.flush()             
    finally:
        agent.eps = eps0

    return EpisodeStats(
        reward=float(mean(rewards)) if rewards else 0.0,
        score=int(mean(scores)) if scores else 0,
        moves=int(mean(moves)) if moves else 0,
        max_tile=int(max(max_tiles) if max_tiles else 0),
    )


# --------------------------------- I/O utils --------------------------------

def prepare_metrics_writer(path: str, append: bool):
    if not path:
        return None, None
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and dest.exists() else "w"
    need_header = (not dest.exists()) or (dest.stat().st_size == 0) or mode == "w"
    f = dest.open(mode, newline="")
    w = csv.writer(f)
    if need_header:
        w.writerow(["episode", "score", "reward", "moves", "max_tile", "epsilon"])
    return w, f


def save_checkpoint(path: str, agent: DQNAgent, buffer: ReplayBuffer | None) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"agent": agent.state_dict(), "replay": buffer.state_dict() if buffer is not None else None}
    with p.open("wb") as h:
        pickle.dump(payload, h, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
    with p.open("rb") as h:
        return pickle.load(h)


def export_model_weights_csv(model: keras.Model, path: str) -> None:
    if not path:
        return
    d = Path(path)
    d.parent.mkdir(parents=True, exist_ok=True)
    with d.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["variable", "flat_index", "value"])
        for var in model.trainable_weights:
            vals = tf.convert_to_tensor(var).numpy().ravel()
            name = var.name
            for idx, value in enumerate(vals):
                w.writerow([name, idx, f"{float(value):.10f}"])


# ---------------------------------- CLI -------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=float, default=0.998)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--step-penalty", type=float, default=0.0)
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--reward-clip", type=float, default=0.0)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--save-path", type=str, default="")
    p.add_argument("--load-path", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--target-sync", type=int, default=1_000)
    p.add_argument("--updates-per-step", type=float, default=0.5)
    p.add_argument("--device", type=str, default="/CPU:0")

    p.add_argument("--channels", type=int, default=128, help="Conv channels per layer")
    p.add_argument("--aug-rot-prob", type=float, default=0.5, help="Prob. of applying a random 90deg rotation to each sample in a batch")
    p.add_argument("--n-step", type=int, default=3, help="N for n-step returns (built offline per episode)")
    p.add_argument("--per-pos-frac", type=float, default=0.5, help="Fraction of batch drawn from positive-reward transitions")

    p.add_argument("--metrics-path", type=str, default="")
    p.add_argument("--metrics-append", action="store_true")
    p.add_argument(
        "--eval-metrics-path",
        type=str,
        default="",
        help="Optional path to a CSV file where per-eval-episode metrics are written",
    )
    p.add_argument(
        "--eval-metrics-append",
        action="store_true",
        help="Append to an existing eval metrics file instead of overwriting it",
    )
    p.add_argument("--weights-csv", type=str, default="")
    return p.parse_args()

# ---------------------------------- Main ------------------------------------

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    env = Game2048()

    agent = DQNAgent(
        n_actions=4,
        channels=args.channels,
        gamma=args.gamma,
        lr=args.lr,
        eps_start=args.epsilon,
        eps_end=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        target_sync=args.target_sync,
        device=args.device,
        seed=args.seed,
    )

    buffer = ReplayBuffer(args.buffer_size, pos_frac=args.per_pos_frac)

    if args.load_path:
        ckpt = load_checkpoint(args.load_path)
        agent.load_state_dict(ckpt.get("agent"))
        replay_state = ckpt.get("replay")
        if replay_state:
            buffer.load_state(replay_state)
        print(f"Loaded checkpoint from '{args.load_path}'")
        # ensure we don't resume with a tiny epsilon
        if agent.eps < max(0.3, args.epsilon_min):
            agent.eps = max(0.3, args.epsilon_min)
            print(f"Reset epsilon to {agent.eps:.3f} for resumed training")

    metrics_writer, metrics_file = prepare_metrics_writer(args.metrics_path, args.metrics_append)

    recent: Deque[EpisodeStats] = deque(maxlen=args.log_interval)

    try:
        for episode in range(1, args.episodes + 1):
            stats = run_episode(
                agent,
                env,
                train=True,
                step_penalty=args.step_penalty,
                buffer=buffer,
                reward_scale=args.reward_scale,
                reward_clip=(args.reward_clip if args.reward_clip > 0 else None),
                nstep=max(1, args.n_step),
                gamma=args.gamma,
            )
            recent.append(stats)

            if metrics_writer:
                metrics_writer.writerow([episode, stats.score, stats.reward, stats.moves, stats.max_tile, agent.eps])
                if metrics_file:
                    metrics_file.flush()

            # Learning updates after the episode
            updates = int(max(1, args.updates_per_step * stats.moves))
            gamma_n = args.gamma ** max(1, args.n_step)
            for _ in range(updates):
                if len(buffer) < max(args.warmup, args.batch_size):
                    break
                batch = buffer.sample(args.batch_size)
                _ = agent.update(
                    batch,
                    aug_rot_prob=max(0.0, min(1.0, args.aug_rot_prob)),
                    gamma_n=gamma_n,
                )

            if episode % args.log_interval == 0:
                avg_score = mean(s.score for s in recent)
                avg_reward = mean(s.reward for s in recent)
                best_tile = max(s.max_tile for s in recent)
                print(
                    f"Episode {episode:>5}: avg score={avg_score:8.1f} | avg reward={avg_reward:8.1f} | "
                    f"best tile={best_tile:4d} | epsilon={agent.eps:.3f}"
                )

        if args.eval_episodes > 0:
            eval_writer, eval_file = prepare_metrics_writer(args.eval_metrics_path, args.eval_metrics_append) if args.eval_metrics_path else (None, None)
            eval_stats = evaluate_agent(
                agent,
                env,
                episodes=args.eval_episodes,
                reward_scale=args.reward_scale,
                reward_clip=(args.reward_clip if args.reward_clip > 0 else None),
                eval_writer=eval_writer,
                eval_file=eval_file,
            )
            if eval_file:
                eval_file.close()
            print(
                f"[Evaluation] avg score={eval_stats.score:.1f} | avg reward={eval_stats.reward:.1f} | "
                f"avg moves={eval_stats.moves:.1f} | best tile={eval_stats.max_tile}"
            )

    finally:
        if metrics_file:
            metrics_file.close()
        if args.save_path:
            save_checkpoint(args.save_path, agent, buffer)
            print(f"Saved checkpoint to '{args.save_path}'")
        if args.weights_csv:
            export_model_weights_csv(agent.q, args.weights_csv)
            print(f"Exported Q-network weights to '{args.weights_csv}'")


if __name__ == "__main__":
    main()
