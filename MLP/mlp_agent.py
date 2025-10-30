"""Train a simple Q-learning agent on the local 2048 environment."""

from __future__ import annotations

# tensorflow used for building the Deep Network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import argparse
import os
import pickle
import random
import csv
import numpy as np

from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Dict, List, Sequence, Tuple

from game2048 import Game2048
from features import encode_state_with_features, shaping_reward


Action = int
StateKey = Tuple[int, ...]


def flatten_state(board: np.ndarray) -> StateKey:
    """Convert the board to a hashable representation for the Q-table."""
    return tuple(int(value) for value in board.flatten())

def encode_state(board: np.ndarray) -> np.ndarray:
    b = board.copy()
    b[b == 0] = 1
    return (np.log2(b).astype(np.float32) - 1.0).ravel()       # shape (16,)

@dataclass
class EpisodeStats:
    reward: float
    score: int
    moves: int
    max_tile: int


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, done, next_avail) -> None:
        # store next available actions as a tuple for easy batching
        self.buf.append((s, a, r, s2, done, tuple(next_avail)))
    def __len__(self) -> int:
        return len(self.buf)
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d, next_avail = zip(*batch)
        return (np.stack(s).astype(np.float32),
                np.array(a, dtype=np.int32),
                np.array(r, dtype=np.float32),
                np.stack(s2).astype(np.float32),
                np.array(d, dtype=np.bool_),
                next_avail)
    
def build_qnet(input_dim: int, hidden: int = 256, n_actions: int = 4) -> keras.Model:
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.Dense(hidden, activation="relu")(x)
    out = layers.Dense(n_actions, activation=None)(x)  # linear Q-values
    return keras.Model(inp, out)

class DQNAgent:
    def __init__(
        self,
        input_dim: int,          # <- NEW: length of feature vector (e.g., ~37)
        n_actions: int = 4,
        hidden: int = 256,       # MLP width (256 -> 256)
        gamma: float = 0.99,
        lr: float = 1e-3,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        target_sync: int = 2000, # steps between target hard updates
        clip_norm: float = 1.0,  # global grad norm clipping
        device: str = "/CPU:0",  # e.g., "/GPU:0"
        seed: int = 42,
    ):
        # --- RNG & basic settings ---
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_sync = target_sync
        self.clip_norm = clip_norm
        self.step_count = 0

        self.input_dim = input_dim
        self.hidden = hidden
        self.device = device

        # Seeds
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # --- Networks, optimizer, loss ---
        with tf.device(device):
            # MLP heads sized to feature vector
            self.q = build_qnet(input_dim=input_dim, hidden=hidden, n_actions=n_actions)
            self.tgt = build_qnet(input_dim=input_dim, hidden=hidden, n_actions=n_actions)
            self.tgt.set_weights(self.q.get_weights())  # hard-sync at start

            self.opt = keras.optimizers.Adam(learning_rate=lr)
            self.loss_fn = keras.losses.Huber()
    
    def select_action(self, state_vec: np.ndarray, available_actions: Sequence[int], explore=True) -> int:
        if not available_actions:
            return 0
        if explore and random.random() < self.eps:
            return random.choice(list(available_actions))
        q = self.q(np.expand_dims(state_vec, 0), training=False).numpy()[0]  # (4,)
        masked = np.full_like(q, -np.inf, dtype=np.float32)
        for a in available_actions:
            masked[a] = q[a]
        best_val = masked.max()
        best_actions = [a for a in available_actions if masked[a] >= best_val - 1e-9]
        return random.choice(best_actions)

    @tf.function
    def _train_step(self, s, a, r, s2, d, next_mask, gamma):
        with tf.GradientTape() as tape:
            q_all = self.q(s, training=True)                   # (B,4)
            q_sa = tf.gather(q_all, a, batch_dims=1)           # (B,)

            q_next_online = self.q(s2, training=False)         # (B,4)
            neg_inf = tf.constant(-1e9, dtype=q_next_online.dtype)
            masked_online = q_next_online + (1.0 - next_mask) * neg_inf
            next_act = tf.argmax(masked_online, axis=1, output_type=tf.int32)

            q_next_target = self.tgt(s2, training=False)       # (B,4)
            q_next_target_sa = tf.gather(q_next_target, next_act, batch_dims=1)

            not_done = tf.cast(tf.logical_not(d), tf.float32)
            target = r + not_done * gamma * q_next_target_sa

            loss = self.loss_fn(target, q_sa)

        grads = tape.gradient(loss, self.q.trainable_variables)
        if self.clip_norm and self.clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss

    def update(self, batch):
        s, a, r, s2, d, next_avail = batch
        B = s.shape[0]
        next_mask = np.zeros((B, self.n_actions), dtype=np.float32)
        for i, avail in enumerate(next_avail):
            if avail:
                next_mask[i, list(avail)] = 1.0

        loss = self._train_step(
            tf.convert_to_tensor(s, dtype=tf.float32),
            tf.convert_to_tensor(a, dtype=tf.int32),
            tf.convert_to_tensor(r, dtype=tf.float32),
            tf.convert_to_tensor(s2, dtype=tf.float32),
            tf.convert_to_tensor(d, dtype=tf.bool),
            tf.convert_to_tensor(next_mask, dtype=tf.float32),
            tf.convert_to_tensor(self.gamma, dtype=tf.float32),
        )
        self.step_count += 1
        if self.step_count % self.target_sync == 0:
            self.tgt.set_weights(self.q.get_weights())
        return float(loss.numpy())

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        
def run_episode(
    agent,                 # DQNAgent (TF)
    env: Game2048,
    *,
    train: bool,
    step_penalty: float,
    buffer=None,           # ReplayBuffer | None
    use_shaping: bool = True,
) -> EpisodeStats:
    """
    Plays one episode. Uses engineered features for state vectors,
    supports optional shaping reward, and records transitions to replay.
    """
    env.reset()

    # initial state vector (needs available actions for features)
    avail = env.get_available_actions()
    s_vec = encode_state_with_features(env.board, avail)

    total_reward = 0.0
    moves = 0
    max_tile = int(env.board.max())

    while True:
        available_actions = env.get_available_actions()
        if not available_actions:
            break

        # epsilon-greedy with illegal-action masking handled inside agent.select_action
        action = agent.select_action(s_vec, available_actions, explore=train)

        prev_score = env.score
        board, score, done = env.step(action)

        # base reward: score delta + step penalty
        reward = float(score - prev_score) + step_penalty

        # optional small heuristic shaping (kept small to not dominate score)
        next_avail = env.get_available_actions()
        if use_shaping:
            reward += shaping_reward(board, next_avail)

        total_reward += reward
        moves += 1
        max_tile = max(max_tile, int(board.max()))

        # next state vector (needs next state's available actions)
        s2_vec = encode_state_with_features(board, next_avail)

        # store transition
        if train and buffer is not None:
            buffer.push(s_vec, action, reward, s2_vec, done, next_avail)

        s_vec = s2_vec
        if done:
            break

    if train:
        agent.decay_epsilon()

    return EpisodeStats(reward=total_reward, score=int(env.score), moves=moves, max_tile=max_tile)

def evaluate_agent(agent: DQNAgent, env: Game2048, episodes: int) -> EpisodeStats:
    scores: List[int] = []
    rewards: List[float] = []
    moves: List[int] = []
    max_tiles: List[int] = []

    original_eps = agent.eps
    agent.eps = 0.0
    try:
        for _ in range(episodes):
            stats = run_episode(agent, env, train=False, step_penalty=0.0, buffer=None)
            scores.append(stats.score)
            rewards.append(stats.reward)
            moves.append(stats.moves)
            max_tiles.append(stats.max_tile)
    finally:
        agent.eps = original_eps

    return EpisodeStats(
        reward=float(mean(rewards)) if rewards else 0.0,
        score=int(mean(scores)) if scores else 0,
        moves=int(mean(moves)) if moves else 0,
        max_tile=int(max(max_tiles) if max_tiles else 0),
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate for Q-updates")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Multiplicative epsilon decay")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum exploration rate")
    parser.add_argument("--step-penalty", type=float, default=0.0, help="Optional penalty applied to every step")
    parser.add_argument("--log-interval", type=int, default=100, help="Episodes between log outputs")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of greedy evaluation runs after training")
    parser.add_argument("--save-path", type=str, default="bin/q_table.pkl", help="Path to store the trained Q-table")
    parser.add_argument("--load-path", type=str, default="", help="Optional path to an existing Q-table to continue training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="",
        help="Optional path to a CSV file where per-episode metrics are written",
    )
    parser.add_argument(
        "--metrics-append",
        action="store_true",
        help="Append to an existing metrics file instead of overwriting it",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate for DQN")
    parser.add_argument("--buffer-size", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=512, help="Replay batch size")
    parser.add_argument("--warmup", type=int, default=5_000, help="Min transitions before learning")
    parser.add_argument("--target-sync", type=int, default=2_000, help="Steps between target net syncs")
    parser.add_argument("--updates-per-step", type=float, default=1.0, help="Gradient updates per env step")
    parser.add_argument("--device", type=str, default="/CPU:0", help="TF device, e.g., /GPU:0")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    env = Game2048()

    # compute input_dim from your feature encoder
    init_avail = env.get_available_actions()
    init_vec = encode_state_with_features(env.board, init_avail)
    input_dim = int(init_vec.shape[0])  # ~37 with current features

    agent = DQNAgent(
        input_dim=input_dim,
        n_actions=4,
        hidden=256,
        gamma=args.gamma,
        lr=args.lr,
        eps_start=args.epsilon,
        eps_end=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        target_sync=args.target_sync,
        device=args.device,
        seed=args.seed,
    )
    buffer = ReplayBuffer(args.buffer_size)

    recent_stats: Deque[EpisodeStats] = deque(maxlen=args.log_interval)

    # --- CSV section: keep exactly as you already implemented ---
    # (no changes needed, omit here for brevity)
    # ------------------------------------------------------------

    try:
        for episode in range(1, args.episodes + 1):
            stats = run_episode(agent, env, train=True, step_penalty=args.step_penalty, buffer=buffer)
            recent_stats.append(stats)

            # metrics_writer block: unchanged

            # learning updates after the episode (you can also do them online per step if you prefer)
            updates = int(max(1, args.updates_per_step * stats.moves))
            for _ in range(updates):
                if len(buffer) < max(args.warmup, args.batch_size):
                    break
                batch = buffer.sample(args.batch_size)
                _ = agent.update(batch)

            if episode % args.log_interval == 0:
                avg_score = mean(stat.score for stat in recent_stats)
                avg_reward = mean(stat.reward for stat in recent_stats)
                best_tile = max(stat.max_tile for stat in recent_stats)
                print(
                    f"Episode {episode:>5}: avg score={avg_score:8.1f} | avg reward={avg_reward:8.1f} | "
                    f"best tile={best_tile:4d} | epsilon={agent.eps:.3f}"
                )
    finally:
        # close metrics file as you already do
        ...

if __name__ == "__main__":
    main()



"""
class QLearningAgent:
    A vanilla Q-learning agent with a tabular value function.

    def __init__(
        self,
        gamma: float = 0.99,        
        alpha: float = 0.1,         # Learning rate
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: Dict[StateKey, np.ndarray] = {}

    def ensure_state(self, state_key: StateKey) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(4, dtype=np.float32)
        return self.q_table[state_key]

    def select_action(self, state_key: StateKey, available_actions: Sequence[Action], explore: bool = True) -> Action:
        if not available_actions:
            return 0

        q_values = self.ensure_state(state_key)

        if explore and random.random() < self.epsilon:
            return random.choice(list(available_actions))

        best_value = max(q_values[action] for action in available_actions)
        best_actions = [action for action in available_actions if q_values[action] >= best_value - 1e-9]
        return random.choice(best_actions)

    def update(self, state_key: StateKey, action: Action, reward: float, next_state_key: StateKey, done: bool) -> None:
        q_values = self.ensure_state(state_key)
        target = reward
        if not done:
            next_q = self.ensure_state(next_state_key)
            target += self.gamma * float(np.max(next_q))
        q_values[action] += self.alpha * (target - q_values[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "gamma": self.gamma,
                    "alpha": self.alpha,
                    "epsilon": self.epsilon,
                    "epsilon_decay": self.epsilon_decay,
                    "epsilon_min": self.epsilon_min,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        self.q_table = data.get("q_table", {})
        self.gamma = float(data.get("gamma", self.gamma))
        self.alpha = float(data.get("alpha", self.alpha))
        self.epsilon = float(data.get("epsilon", self.epsilon))
        self.epsilon_decay = float(data.get("epsilon_decay", self.epsilon_decay))
        self.epsilon_min = float(data.get("epsilon_min", self.epsilon_min))
"""