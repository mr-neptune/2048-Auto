"""
Record whole games (every board, move and score) as JSON files for the web page.

Games are played with the same code and random seed as the 500-game tests, so the look-ahead run
reproduces those test games; compare its --metrics-path CSV with the test's CSV to confirm. A stopped
run continues from the next game when started again.

  python -m CNN.record_games --method afterstate_search --games 500 --out-dir site/games \
      --metrics-path CNN/logs/record_afterstate_search.csv
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from game2048 import Game2048
from CNN.cnn_agent import DQNAgent, board_to_onehot, prepare_metrics_writer
from CNN.cnn_agent import load_checkpoint as load_dqn_checkpoint
from CNN.cnn_afterstate import (AfterstateAgent, EpisodeStats, board_to_exp, load_checkpoint, resume_eval, run_episode,
                                run_search_episode, save_eval_progress)

METHODS = ("random", "cnn", "afterstate", "afterstate_search")
REWARD_SCALE = 1e-3


class RecordingGame(Game2048):
    """Game2048 that keeps every board, move and score of the current game."""

    def reset(self):
        super().reset()
        self.boards, self.actions, self.scores = [self.board.copy()], [], [int(self.score)]

    def step(self, action):
        result = super().step(action)
        self.boards.append(self.board.copy())
        self.actions.append(int(action))
        self.scores.append(int(self.score))
        return result


def build_player(args):
    if args.method in ("afterstate", "afterstate_search"):
        ckpt = load_checkpoint(args.afterstate_ckpt)
        a = ckpt["args"]
        agent = AfterstateAgent(channels=a.get("channels", 128), hidden=a.get("hidden", 256), gamma=a.get("gamma", 1.0),
                                arch=a.get("arch", "v1"), seed=args.seed)
        agent.load_state_dict(ckpt["agent"])
        return agent
    if args.method == "cnn":
        agent = DQNAgent(n_actions=4, channels=128, gamma=0.99, lr=1e-4, eps_start=0.0, eps_end=0.0, eps_decay=1.0,
                         target_sync=1000, device="/CPU:0", seed=args.seed)
        agent.load_state_dict(load_dqn_checkpoint(args.dqn_ckpt).get("agent"))
        agent.eps = 0.0
        return agent
    return None


def play_game(method: str, agent, env: RecordingGame) -> EpisodeStats:
    if method == "afterstate_search":
        return run_search_episode(agent, env, reward_scale=REWARD_SCALE)
    if method == "afterstate":
        return run_episode(agent, env, reward_scale=REWARD_SCALE, epsilon=0.0)[0]
    env.reset()
    moves = 0
    while True:
        available = env.get_available_actions()
        if not available:
            break
        if method == "random":
            action = random.choice(available)
        else:
            action = int(agent.select_action(board_to_onehot(env.board), available, explore=False))
        _, _, done = env.step(action)
        moves += 1
        if done:
            break
    return EpisodeStats(reward=REWARD_SCALE * env.score, score=int(env.score), moves=moves, max_tile=int(env.board.max()))


def write_game(path: Path, env: RecordingGame, stats: EpisodeStats) -> None:
    """boards: 16 hex digits per board (tile exponents, row by row); gains: points earned by each move."""
    payload = {
        "score": stats.score,
        "moves": stats.moves,
        "max_tile": stats.max_tile,
        "actions": "".join(str(a) for a in env.actions),
        "boards": "".join(f"{e:x}" for b in env.boards for e in board_to_exp(b).flat),
        "gains": [b - a for a, b in zip(env.scores, env.scores[1:])],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--out-dir", required=True, help="Each game is written to <out-dir>/<method>/<game number>.json")
    p.add_argument("--metrics-path", required=True, help="One result row per game; also where a stopped run resumes from")
    p.add_argument("--afterstate-ckpt", default="bin/cnn_after_v2lr20_ep20000.chkpt")
    p.add_argument("--dqn-ckpt", default="bin/cnn_v3_09.chkpt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0, help="Cap TensorFlow CPU threads (0 = use all cores)")
    args = p.parse_args()

    tf.config.set_visible_devices([], "GPU")  # the Metal GPU returns wrong board values
    if args.threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(min(2, args.threads))

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = RecordingGame()
    agent = build_player(args)

    game_dir = Path(args.out_dir) / args.method
    game_dir.mkdir(parents=True, exist_ok=True)
    run = {"method": args.method, "seed": args.seed}
    done = len(resume_eval(args.metrics_path, run))
    writer, handle = prepare_metrics_writer(args.metrics_path, append=done > 0)
    for g in range(done + 1, args.games + 1):
        stats = play_game(args.method, agent, env)
        write_game(game_dir / f"{g}.json", env, stats)
        writer.writerow([g, stats.score, stats.reward, stats.moves, stats.max_tile, 0.0])
        handle.flush()
        save_eval_progress(args.metrics_path, run, g)
        if g % 10 == 0:
            print(f"game {g}/{args.games}: score {stats.score}, max tile {stats.max_tile}", flush=True)
    handle.close()
    print(f"Recorded {args.games} {args.method} games in '{game_dir}'")


if __name__ == "__main__":
    main()
