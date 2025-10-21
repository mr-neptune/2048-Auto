"""Train a simple Q-learning agent on the local 2048 environment."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import csv
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np

from game2048 import Game2048


Action = int
StateKey = Tuple[int, ...]


def flatten_state(board: np.ndarray) -> StateKey:
    """Convert the board to a hashable representation for the Q-table."""
    return tuple(int(value) for value in board.flatten())


@dataclass
class EpisodeStats:
    reward: float
    score: int
    moves: int
    max_tile: int


class QLearningAgent:
    """A vanilla Q-learning agent with a tabular value function."""

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


def run_episode(
    agent: QLearningAgent,
    env: Game2048,
    *,
    train: bool,
    step_penalty: float,
) -> EpisodeStats:
    env.reset()
    state_key = flatten_state(env.board)

    total_reward = 0.0
    moves = 0
    max_tile = int(env.board.max())

    while True:
        available_actions = env.get_available_actions()
        if not available_actions:
            break

        action = agent.select_action(state_key, available_actions, explore=train)
        previous_score = env.score

        board, score, game_over = env.step(action)
        reward = float(score - previous_score) + step_penalty

        total_reward += reward
        moves += 1
        max_tile = max(max_tile, int(board.max()))

        next_state_key = flatten_state(board)
        if train:
            agent.update(state_key, action, reward, next_state_key, game_over)
        state_key = next_state_key

        if game_over:
            break

    if train:
        agent.decay_epsilon()

    return EpisodeStats(reward=total_reward, score=int(env.score), moves=moves, max_tile=max_tile)


def evaluate_agent(agent: QLearningAgent, env: Game2048, episodes: int) -> EpisodeStats:
    scores: List[int] = []
    rewards: List[float] = []
    moves: List[int] = []
    max_tiles: List[int] = []

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        for _ in range(episodes):
            stats = run_episode(agent, env, train=False, step_penalty=0.0)
            scores.append(stats.score)
            rewards.append(stats.reward)
            moves.append(stats.moves)
            max_tiles.append(stats.max_tile)
    finally:
        agent.epsilon = original_epsilon

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    agent = QLearningAgent(
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    if args.load_path:
        agent.load(args.load_path)

    env = Game2048()
    recent_stats: Deque[EpisodeStats] = deque(maxlen=args.log_interval)

    metrics_file = None
    metrics_writer = None
    if args.metrics_path:
        metrics_dir = os.path.dirname(args.metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        open_mode = "a" if args.metrics_append else "w"
        should_write_header = True
        if args.metrics_append and os.path.exists(args.metrics_path):
            try:
                should_write_header = os.path.getsize(args.metrics_path) == 0
            except OSError:
                should_write_header = True
        metrics_file = open(args.metrics_path, open_mode, newline="", encoding="utf-8")
        fieldnames = ["episode", "score", "reward", "moves", "max_tile"]
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        if not args.metrics_append or should_write_header:
            metrics_writer.writeheader()
            metrics_file.flush()

    try:
        for episode in range(1, args.episodes + 1):
            stats = run_episode(agent, env, train=True, step_penalty=args.step_penalty)
            recent_stats.append(stats)

            if metrics_writer and metrics_file:
                metrics_writer.writerow(
                    {
                        "episode": episode,
                        "score": stats.score,
                        "reward": stats.reward,
                        "moves": stats.moves,
                        "max_tile": stats.max_tile,
                    }
                )
                metrics_file.flush()

            if episode % args.log_interval == 0:
                avg_score = mean(stat.score for stat in recent_stats)
                avg_reward = mean(stat.reward for stat in recent_stats)
                best_tile = max(stat.max_tile for stat in recent_stats)
                print(
                    f"Episode {episode:>5}: avg score={avg_score:8.1f} | avg reward={avg_reward:8.1f} | "
                    f"best tile={best_tile:4d} | epsilon={agent.epsilon:.3f}"
                )
    finally:
        if metrics_file:
            metrics_file.flush()
            metrics_file.close()

    if args.save_path:
        agent.save(args.save_path)
        print(f"Saved Q-table to {args.save_path}")

    if args.eval_episodes > 0:
        eval_stats = evaluate_agent(agent, env, args.eval_episodes)
        print(
            "Evaluation (greedy policy): "
            f"avg score={eval_stats.score} | avg reward={eval_stats.reward:.1f} | "
            f"avg moves={eval_stats.moves} | best tile={eval_stats.max_tile}"
        )


if __name__ == "__main__":
    main()

