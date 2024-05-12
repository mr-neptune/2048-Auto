from flask import Flask, jsonify, request, render_template
from game2048 import Game2048  # Ensure this matches your game logic file
import numpy as np

app = Flask(__name__)
game = Game2048()  # Initialize your game here

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

if __name__ == '__main__':
    app.run(debug=True)
