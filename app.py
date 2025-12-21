import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, render_template

from CNN.cnn_agent import DQNAgent, board_to_onehot, load_checkpoint
from game2048 import Game2048

app = Flask(__name__)
game = Game2048()  # Initialize game here

# ------------------------- CNN agent (pure eval) -------------------------
CNN_CKPT_PATH = os.environ.get("CNN_CKPT_PATH", "bin/cnn_v3_03.chkpt")
CNN_CHANNELS = int(os.environ.get("CNN_CHANNELS", "128"))
CNN_DEVICE = os.environ.get("CNN_DEVICE", "/CPU:0")

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
    agent = get_cnn_agent()
    if agent is None:
        return jsonify(error=f"CNN checkpoint not found at '{CNN_CKPT_PATH}'."), 404

    game.reset()
    board = [[int(cell) for cell in row] for row in game.board.tolist()]
    return jsonify(board=board, score=int(game.score), game_over=False)


@app.route('/agent/move', methods=['POST'])
def agent_move():
    agent = get_cnn_agent()
    if agent is None:
        return jsonify(error="CNN agent is not loaded."), 404

    available_actions = game.get_available_actions()
    if not available_actions:
        board = [[int(cell) for cell in row] for row in game.board.tolist()]
        return jsonify(board=board, score=int(game.score), game_over=True, action=None)

    state = board_to_onehot(game.board)
    action = int(agent.select_action(state, available_actions, explore=False))

    board, score, game_over = game.step(action)
    board_python = [[int(cell) for cell in row] for row in board.tolist()]
    return jsonify(board=board_python, score=int(score), game_over=game_over, action=action)

if __name__ == '__main__':
    app.run(debug=True)
