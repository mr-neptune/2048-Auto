import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, render_template

from CNN.cnn_agent import DQNAgent, board_to_onehot, load_checkpoint
from CNN.cnn_afterstate import AfterstateAgent, board_to_exp, legal_afterstates
from CNN.afterstate_search import expectimax_action
from game2048 import Game2048

app = Flask(__name__)
game = Game2048()  # Initialize game here

# ------------------------- CNN agent (pure eval) -------------------------
CNN_CKPT_PATH = os.environ.get("CNN_CKPT_PATH", "bin/cnn_v3_09.chkpt")
CNN_CHANNELS = int(os.environ.get("CNN_CHANNELS", "128"))
CNN_DEVICE = os.environ.get("CNN_DEVICE", "/GPU:0")

_cnn_agent = None


def get_cnn_agent():
    """Lazily load the CNN agent for pure eval; epsilon forced to zero."""
    global _cnn_agent
    if _cnn_agent is not None:
        return _cnn_agent

    ckpt_path = Path(CNN_CKPT_PATH)
    if not ckpt_path.exists():
        return None

    agent = DQNAgent(
        n_actions=4,
        channels=CNN_CHANNELS,
        gamma=0.99,
        lr=1e-4,
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=1.0,
        target_sync=1000,
        device=CNN_DEVICE,
        seed=42,
    )
    state = load_checkpoint(str(ckpt_path))
    agent.load_state_dict(state.get("agent"))
    agent.eps = 0.0
    _cnn_agent = agent
    return _cnn_agent


# ------------------- CNN afterstate agent (pure eval) --------------------
AFTERSTATE_CKPT_PATH = os.environ.get("AFTERSTATE_CKPT_PATH", "bin/cnn_after_pilot.chkpt")

_afterstate_agent = None
_afterstate_reward_scale = 1e-3


def get_afterstate_agent():
    """Lazily load the afterstate value agent; network shape comes from the checkpoint's args."""
    global _afterstate_agent, _afterstate_reward_scale
    if _afterstate_agent is not None:
        return _afterstate_agent

    ckpt_path = Path(AFTERSTATE_CKPT_PATH)
    if not ckpt_path.exists():
        return None

    state = load_checkpoint(str(ckpt_path))
    args = state.get("args", {})
    agent = AfterstateAgent(
        channels=args.get("channels", 128),
        hidden=args.get("hidden", 256),
        gamma=args.get("gamma", 1.0),
    )
    agent.load_state_dict(state.get("agent"))
    _afterstate_reward_scale = args.get("reward_scale", 1e-3)
    _afterstate_agent = agent
    return _afterstate_agent


AFTERSTATE_MODELS = ("afterstate", "afterstate_search")


def get_agent_for(model):
    if model in AFTERSTATE_MODELS:
        return get_afterstate_agent(), AFTERSTATE_CKPT_PATH
    return get_cnn_agent(), CNN_CKPT_PATH


def requested_model():
    return (request.get_json(silent=True) or {}).get("model", "cnn")

@app.route('/')
def home():
    return render_template('index.html')  # This should be the path to your HTML file

@app.route('/init', methods=['GET'])
def init_game():
    # Reset the game to the initial state
    game.reset()
    # Convert the NumPy array to a Python list and np.int64 to int
    board = [[int(cell) for cell in row] for row in game.board.tolist()]
    return jsonify(board=board, score=int(game.score), game_over=False)

@app.route('/move', methods=['POST'])
def move():
    action = request.json['action']
    board, score, game_over = game.step(action)

    # Convert the board to a Python list and np.int64 to int
    board_python = [[int(cell) for cell in row] for row in board.tolist()]
    return jsonify(board=board_python, score=int(score), game_over=game_over)


@app.route('/agent/init', methods=['POST'])
def agent_init():
    agent, ckpt_path = get_agent_for(requested_model())
    if agent is None:
        return jsonify(error=f"Checkpoint not found at '{ckpt_path}'."), 404

    game.reset()
    board = [[int(cell) for cell in row] for row in game.board.tolist()]
    return jsonify(board=board, score=int(game.score), game_over=False)


def choose_afterstate_action(agent):
    """Greedy afterstate move for the current board, or None when no move is legal."""
    actions, afterstates, deltas = legal_afterstates(game)
    if afterstates is None or deltas is None:
        return None
    i, _ = agent.select(afterstates, _afterstate_reward_scale * deltas)
    return int(actions[i])


def choose_search_action(agent):
    """2-ply expectimax move (looks one tile spawn ahead), or None when no move is legal."""
    return expectimax_action(agent, board_to_exp(game.board), _afterstate_reward_scale)


def choose_dqn_action(agent):
    """Greedy DQN move for the current board, or None when no move is legal."""
    available_actions = game.get_available_actions()
    if not available_actions:
        return None
    state = board_to_onehot(game.board)
    return int(agent.select_action(state, available_actions, explore=False))


@app.route('/agent/move', methods=['POST'])
def agent_move():
    model = requested_model()
    agent, _ = get_agent_for(model)
    if agent is None:
        return jsonify(error="Agent is not loaded."), 404

    if model == "afterstate_search":
        action = choose_search_action(agent)
    elif model == "afterstate":
        action = choose_afterstate_action(agent)
    else:
        action = choose_dqn_action(agent)
    if action is None:
        board = [[int(cell) for cell in row] for row in game.board.tolist()]
        return jsonify(board=board, score=int(game.score), game_over=True, action=None)

    board, score, game_over = game.step(action)
    board_python = [[int(cell) for cell in row] for row in board.tolist()]
    return jsonify(board=board_python, score=int(score), game_over=game_over, action=action)

if __name__ == '__main__':
    app.run(debug=True)
