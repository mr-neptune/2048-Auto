from __future__ import annotations

"""
Afterstate value-learning agent for 2048 (same directional CNN trunk as cnn_agent.py).

Instead of Q(s, a), learn V(afterstate): the board right after a slide, before the
random tile spawns. The slide is deterministic, so
    action = argmax_a [ r(s, a) + gamma * V(afterstate(s, a)) ]
and spawn randomness never enters action selection. This is how the strong 2048
learners (n-tuple TD players, CNN value nets) are trained.

 - Reward = raw game score delta (scaled); no shaping
 - Greedy play (epsilon 0 by default): random spawns provide exploration
 - TD(lambda) returns built at episode end from values computed while acting
 - Replay buffer of (afterstate, return) with all 8 board symmetries as augmentation
   (V is symmetry-invariant, so no action remapping is needed)
 - Atomic periodic checkpoints and --resume, so interrupted runs continue
"""

import argparse
import math
import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from game2048 import Game2048
from CNN.cnn_agent import N_CHANNELS, conv_block_h, conv_block_v, cross_maps, prepare_metrics_writer
from CNN.afterstate_search import expectimax_action

TILE_THRESHOLDS = (512, 1024, 2048, 4096, 8192, 16384)


# ------------------------------- Encoding -----------------------------------

def board_to_exp(board: np.ndarray) -> np.ndarray:
    """Return (4,4) uint8 exponents: 0=empty, k=2^k (clamped to N_CHANNELS-1)."""
    b = np.asarray(board)
    e = np.zeros(b.shape, dtype=np.uint8)
    nz = b > 0
    e[nz] = np.minimum(np.log2(b[nz]), N_CHANNELS - 1).astype(np.uint8)
    return e


def random_symmetry(exps: np.ndarray) -> np.ndarray:
    """Apply one of the 8 board symmetries (4 rotations x mirror) to each board in a batch."""
    ks = np.random.randint(0, 8, size=len(exps))
    out = exps.copy()
    for k in range(1, 8):
        idx = np.flatnonzero(ks == k)
        if idx.size:
            x = exps[idx]
            if k >= 4:
                x = x[:, :, ::-1]
            out[idx] = np.rot90(x, k % 4, axes=(1, 2))
    return out


def legal_afterstates(env: Game2048):
    """Return (actions, afterstate exponents (k,4,4), raw score deltas (k,)) for legal moves."""
    actions, boards, deltas = [], [], []
    for a in range(4):
        board, delta, changed = env.peek_move(a)
        if changed:
            actions.append(a)
            boards.append(board_to_exp(board))
            deltas.append(delta)
    if not actions:
        return actions, None, None
    return actions, np.stack(boards), np.array(deltas, dtype=np.float32)


# ------------------------------- Replay -------------------------------------

class ReturnBuffer:
    """Ring buffer of (afterstate exponents, TD(lambda) return)."""
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.exps = np.zeros((capacity, 4, 4), dtype=np.uint8)
        self.targets = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push_many(self, exps: np.ndarray, targets: np.ndarray) -> None:
        for e, g in zip(exps, targets):
            self.exps[self.pos] = e
            self.targets[self.pos] = g
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return random_symmetry(self.exps[idxs]), self.targets[idxs]

    def state_dict(self):
        n = self.size
        return {"capacity": self.capacity, "pos": self.pos, "size": n,
                "exps": self.exps[:n], "targets": self.targets[:n]}

    def load_state(self, state) -> None:
        if not state:
            return
        n = min(int(state["size"]), self.capacity)
        self.exps[:n] = state["exps"][:n]
        self.targets[:n] = state["targets"][:n]
        self.size = n
        self.pos = int(state["pos"]) % self.capacity


def lambda_returns(rewards, values, gamma: float, lam: float) -> np.ndarray:
    """
    Target for afterstate t: G_t = r_{t+1} + gamma * ((1-lam) * V_{t+1} + lam * G_{t+1}).
    The last afterstate leads to game over, so G_{T-1} = 0.
    """
    T = len(rewards)
    G = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in range(T - 2, -1, -1):
        g = rewards[t + 1] + gamma * ((1.0 - lam) * values[t + 1] + lam * g)
        G[t] = g
    return G


# ------------------------------ Value network -------------------------------

def build_vnet(channels: int = 128, hidden: int = 256, arch: str = "v1") -> keras.Model:
    """
    v1: pair/triple/2x2 directional maps -> Dense(hidden) -> V.
    v2: v1 maps plus full-row (1x4), full-column (4x1) and 2x3 / 3x2 block detectors,
        and a deeper head Dense(2*hidden) -> Dense(hidden) -> V.
    """
    inp = keras.Input(shape=(4, 4, N_CHANNELS))
    h1, h2 = conv_block_h(inp, channels)
    v1, v2 = conv_block_v(inp, channels)
    x1, x2 = cross_maps(h1, v1, channels)
    maps = [h1, h2, v1, v2, x1, x2]
    if arch == "v2":
        maps += [
            layers.Conv2D(channels, (1, 4), padding="valid", activation="relu")(inp),  # 4x1: whole rows
            layers.Conv2D(channels, (4, 1), padding="valid", activation="relu")(inp),  # 1x4: whole columns
            layers.Conv2D(channels, (2, 3), padding="valid", activation="relu")(inp),  # 3x2
            layers.Conv2D(channels, (3, 2), padding="valid", activation="relu")(inp),  # 2x3
        ]
    feats = layers.Concatenate()([layers.Flatten()(t) for t in maps])
    if arch == "v2":
        feats = layers.Dense(2 * hidden, activation="relu")(feats)
    trunk = layers.Dense(hidden, activation="relu")(feats)
    out = layers.Dense(1, activation=None)(trunk)
    return keras.Model(inp, out)


class AfterstateAgent:
    def __init__(self, channels: int = 128, hidden: int = 256, lr: float = 1e-4, gamma: float = 1.0,
                 clip_norm: float = 1.0, seed: int = 42, arch: str = "v1") -> None:
        self.gamma = gamma
        self.clip_norm = clip_norm
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.v = build_vnet(channels=channels, hidden=hidden, arch=arch)
        self.opt = keras.optimizers.Adam(learning_rate=lr)
        self.opt.build(self.v.trainable_variables)

    @tf.function(input_signature=[tf.TensorSpec((None, 4, 4), tf.uint8)])
    def _values(self, exps):
        x = tf.one_hot(tf.cast(exps, tf.int32), N_CHANNELS)
        return self.v(x, training=False)[:, 0]

    def values(self, exps: np.ndarray) -> np.ndarray:
        return self._values(exps).numpy()

    def select(self, exps: np.ndarray, rewards: np.ndarray, epsilon: float = 0.0):
        """Return (index of chosen afterstate, V of every afterstate)."""
        v = self.values(exps)
        if epsilon > 0 and random.random() < epsilon:
            return random.randrange(len(rewards)), v
        return int(np.argmax(rewards + self.gamma * v)), v

    @tf.function(input_signature=[tf.TensorSpec((None, 4, 4), tf.uint8), tf.TensorSpec((None,), tf.float32)])
    def _train_step(self, exps, targets):
        x = tf.one_hot(tf.cast(exps, tf.int32), N_CHANNELS)
        with tf.GradientTape() as tape:
            pred = self.v(x, training=True)[:, 0]
            loss = tf.reduce_mean(tf.square(targets - pred))
        variables = self.v.trainable_variables
        grads = tape.gradient(loss, variables)
        if self.clip_norm and self.clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.opt.apply_gradients(zip(grads, variables))
        return loss

    def update(self, exps: np.ndarray, targets: np.ndarray) -> float:
        return float(self._train_step(exps, targets))

    def set_lr(self, lr: float) -> None:
        self.opt.learning_rate = lr

    def state_dict(self):
        return {"v_weights": self.v.get_weights(), "opt_vars": [v.numpy() for v in self.opt.variables]}

    def load_state_dict(self, state) -> None:
        if not state:
            return
        self.v.set_weights(state["v_weights"])
        opt_vars = state.get("opt_vars")
        if opt_vars and len(opt_vars) == len(self.opt.variables):
            for var, val in zip(self.opt.variables, opt_vars):
                var.assign(val)


# ------------------------------- Episodes -----------------------------------

@dataclass
class EpisodeStats:
    reward: float
    score: int
    moves: int
    max_tile: int


def run_episode(agent: AfterstateAgent, env: Game2048, *, reward_scale: float, epsilon: float):
    """Play one game; return (stats, chosen afterstates, scaled rewards, V at acting time)."""
    env.reset()
    exps, rewards, values = [], [], []
    while True:
        actions, as_exps, deltas = legal_afterstates(env)
        if not actions:
            break
        r = reward_scale * deltas
        i, v = agent.select(as_exps, r, epsilon)
        _, _, done = env.step(actions[i])
        exps.append(as_exps[i])
        rewards.append(float(r[i]))
        values.append(float(v[i]))
        if done:
            break
    stats = EpisodeStats(reward=float(sum(rewards)), score=int(env.score), moves=len(exps), max_tile=int(env.board.max()))
    return stats, exps, rewards, values


def scheduled_lr(args, episode: int) -> float:
    """Cosine decay from --lr to --lr-min over episodes [--lr-decay-start, --episodes]; constant otherwise."""
    if args.lr_schedule != "cosine":
        return args.lr
    span = max(1, args.episodes - args.lr_decay_start)
    progress = min(1.0, max(0.0, (episode - args.lr_decay_start) / span))
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * progress))


def run_search_episode(agent: AfterstateAgent, env: Game2048, *, reward_scale: float) -> EpisodeStats:
    """Play one game choosing every move by 2-ply expectimax (evaluation only, no training data)."""
    env.reset()
    moves = 0
    while True:
        action = expectimax_action(agent, board_to_exp(env.board), reward_scale)
        if action is None:
            break
        _, _, done = env.step(action)
        moves += 1
        if done:
            break
    return EpisodeStats(reward=reward_scale * env.score, score=int(env.score), moves=moves, max_tile=int(env.board.max()))


def tile_rates(max_tiles) -> dict:
    n = max(1, len(max_tiles))
    return {t: sum(m >= t for m in max_tiles) / n for t in TILE_THRESHOLDS}


def evaluate(agent: AfterstateAgent, env: Game2048, games: int, reward_scale: float, writer=None, handle=None,
             search_depth: int = 1):
    scores, tiles = [], []
    for g in range(1, games + 1):
        if search_depth >= 2:
            stats = run_search_episode(agent, env, reward_scale=reward_scale)
        else:
            stats, *_ = run_episode(agent, env, reward_scale=reward_scale, epsilon=0.0)
        scores.append(stats.score)
        tiles.append(stats.max_tile)
        if writer is not None:
            writer.writerow([g, stats.score, stats.reward, stats.moves, stats.max_tile, 0.0])
            handle.flush()
    print(f"[Evaluation] {games} games | avg score={mean(scores):.0f} | best score={max(scores)}")
    for t, rate in tile_rates(tiles).items():
        print(f"  reached {t:>5}: {rate * 100:6.2f}%")
    return scores, tiles


# --------------------------------- I/O --------------------------------------

def save_checkpoint(path: str, agent: AfterstateAgent, buffer: ReturnBuffer, episode: int, args) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"agent": agent.state_dict(), "replay": buffer.state_dict(), "episode": episode, "args": vars(args)}
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as h:
        pickle.dump(payload, h, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)  # atomic: a crash mid-write never corrupts the last good checkpoint


def snapshot_path(save_path: str, episode: int) -> str:
    p = Path(save_path)
    return str(p.with_name(f"{p.stem}_ep{episode}{p.suffix}"))


def load_checkpoint(path: str) -> dict:
    with Path(path).open("rb") as h:
        return pickle.load(h)


# ---------------------------------- CLI -------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=2000, help="Total episodes (a resumed run continues up to this number)")
    p.add_argument("--save-path", type=str, default="", help="Checkpoint path (also the --resume source)")
    p.add_argument("--resume", action="store_true", help="Continue from --save-path if it exists")
    p.add_argument("--load-path", type=str, default="", help="Load weights only (fresh replay, episode counter from 1)")
    p.add_argument("--checkpoint-interval", type=int, default=200)
    p.add_argument("--snapshot-interval", type=int, default=0,
                   help="Also keep a copy <save-path stem>_ep<N> every N episodes (0 to disable)")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="/CPU:0", help="/CPU:0 is ~2x faster than Metal /GPU:0 on Apple silicon; use /GPU:0 on CUDA")

    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.5, help="TD(lambda): 0 = one-step TD, 1 = Monte Carlo")
    p.add_argument("--reward-scale", type=float, default=1e-3, help="Score delta multiplier (values stay ~O(1-100))")
    p.add_argument("--epsilon", type=float, default=0.0, help="Random-move probability while training")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (peak rate when --lr-schedule cosine)")
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--lr-min", type=float, default=1e-5, help="Final learning rate for --lr-schedule cosine")
    p.add_argument("--lr-decay-start", type=int, default=0, help="Episode where the cosine decay begins")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--arch", choices=("v1", "v2"), default="v1", help="v2 adds row/column/block detectors and a deeper head")
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--replay-ratio", type=float, default=8.0, help="Average number of times each afterstate is trained on")
    p.add_argument("--warmup", type=int, default=5_000)

    p.add_argument("--metrics-path", type=str, default="")
    p.add_argument("--eval-episodes", type=int, default=0, help="Greedy games to play after training")
    p.add_argument("--eval-only", action="store_true", help="Skip training; evaluate --load-path/--save-path")
    p.add_argument("--search-depth", type=int, choices=(1, 2), default=1,
                   help="Evaluation move choice: 1 = greedy, 2 = expectimax over the next tile spawn")
    p.add_argument("--eval-metrics-path", type=str, default="")
    return p.parse_args()


# ---------------------------------- Main ------------------------------------

def main() -> None:
    args = parse_args()
    if args.device.upper().startswith("/CPU"):
        tf.config.set_visible_devices([], "GPU")

    env = Game2048()
    agent = AfterstateAgent(channels=args.channels, hidden=args.hidden, lr=args.lr, gamma=args.gamma, seed=args.seed,
                            arch=args.arch)
    buffer = ReturnBuffer(args.buffer_size)

    start_episode = 1
    if args.resume and args.save_path and Path(args.save_path).exists():
        ckpt = load_checkpoint(args.save_path)
        agent.load_state_dict(ckpt["agent"])
        buffer.load_state(ckpt.get("replay"))
        start_episode = int(ckpt.get("episode", 0)) + 1
        print(f"Resumed from '{args.save_path}' at episode {start_episode}")
    elif args.load_path:
        agent.load_state_dict(load_checkpoint(args.load_path)["agent"])
        print(f"Loaded weights from '{args.load_path}'")

    if args.eval_only:
        writer, handle = prepare_metrics_writer(args.eval_metrics_path, append=False)
        evaluate(agent, env, max(1, args.eval_episodes), args.reward_scale, writer, handle, args.search_depth)
        if handle:
            handle.close()
        return

    metrics_writer, metrics_file = prepare_metrics_writer(args.metrics_path, append=start_episode > 1)
    recent = deque(maxlen=args.log_interval)
    update_carry = 0.0
    losses = deque(maxlen=1000)
    t_log, moves_log = time.time(), 0
    last_done = start_episode - 1  # last fully finished episode (trained + logged)

    try:
        for episode in range(start_episode, args.episodes + 1):
            stats, exps, rewards, values = run_episode(agent, env, reward_scale=args.reward_scale, epsilon=args.epsilon)
            if exps:
                buffer.push_many(np.stack(exps), lambda_returns(rewards, values, args.gamma, args.lam))
            recent.append(stats)
            moves_log += stats.moves

            update_carry += stats.moves * args.replay_ratio / args.batch_size
            n_updates, update_carry = int(update_carry), update_carry - int(update_carry)
            lr = scheduled_lr(args, episode)
            agent.set_lr(lr)
            if len(buffer) >= max(args.warmup, args.batch_size):
                for _ in range(n_updates):
                    losses.append(agent.update(*buffer.sample(args.batch_size)))

            if metrics_writer:
                metrics_writer.writerow([episode, stats.score, stats.reward, stats.moves, stats.max_tile, args.epsilon])
                metrics_file.flush()
            last_done = episode

            if episode % args.log_interval == 0:
                rates = tile_rates([s.max_tile for s in recent])
                dt = time.time() - t_log
                print(
                    f"Episode {episode:>6}: avg score={mean(s.score for s in recent):8.0f} | "
                    f"1024={rates[1024]*100:5.1f}% 2048={rates[2048]*100:5.1f}% 4096={rates[4096]*100:5.1f}% | "
                    f"best tile={max(s.max_tile for s in recent):5d} | "
                    f"loss={mean(losses) if losses else 0:.4f} lr={lr:.2e} | "
                    f"{dt / len(recent):.2f}s/ep {moves_log / max(dt, 1e-9):.0f} moves/s",
                    flush=True,
                )
                t_log, moves_log = time.time(), 0

            if args.save_path and args.checkpoint_interval > 0 and episode % args.checkpoint_interval == 0:
                save_checkpoint(args.save_path, agent, buffer, episode, args)
            if args.save_path and args.snapshot_interval > 0 and episode % args.snapshot_interval == 0:
                save_checkpoint(snapshot_path(args.save_path, episode), agent, buffer, episode, args)

        if args.eval_episodes > 0:
            writer, handle = prepare_metrics_writer(args.eval_metrics_path, append=False)
            evaluate(agent, env, args.eval_episodes, args.reward_scale, writer, handle, args.search_depth)
            if handle:
                handle.close()
    finally:
        if metrics_file:
            metrics_file.close()
        if args.save_path:
            save_checkpoint(args.save_path, agent, buffer, last_done, args)
            print(f"Saved checkpoint at episode {last_done} to '{args.save_path}'")


if __name__ == "__main__":
    main()
