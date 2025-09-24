import numpy as np

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        """Reset the game board."""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_new_tile()
        self.add_new_tile()
    
    def add_new_tile(self):
        """Add a new tile to a random empty space on the board."""
        empty_tiles = np.argwhere(self.board == 0)
        if len(empty_tiles) > 0:
            y, x = empty_tiles[np.random.choice(len(empty_tiles))]
            self.board[y, x] = np.random.choice([2, 4], p=[0.9, 0.1])
    
    def is_game_over(self):
        """Check if the game is over."""
        if 0 in self.board:
            return False
        for i in range(self.size):
            for j in range(self.size):
                if i < self.size - 1 and self.board[i][j] == self.board[i + 1][j]:
                    return False
                if j < self.size - 1 and self.board[i][j] == self.board[i][j + 1]:
                    return False
        return True

    def compress(self, row):
        """Compress the zeroes in a row."""
        new_row = [i for i in row if i != 0]
        new_row += [0] * (self.size - len(new_row))
        return new_row

    def merge(self, row):
        """Merge the row and calculate score."""
        for j in range(self.size - 1):
            if row[j] == row[j + 1] and row[j] != 0:
                row[j] *= 2
                row[j + 1] = 0
                self.score += row[j]
        return row

    def reverse(self, row):
        """Reverse the row."""
        return row[::-1]

    def transpose(self):
        """Transpose the board."""
        self.board = np.transpose(self.board)

    def move_left(self):
        """Move tiles left."""
        board_changed = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            self.board[i] = self.compress(self.board[i])
            self.board[i] = self.merge(self.board[i])
            self.board[i] = self.compress(self.board[i])
            if not all(original_row == self.board[i]):
                board_changed = True
        return board_changed

    def move_right(self):
        """Move tiles right."""
        board_changed = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            self.board[i] = self.reverse(self.board[i])
            self.board[i] = self.compress(self.board[i])
            self.board[i] = self.merge(self.board[i])
            self.board[i] = self.compress(self.board[i])
            self.board[i] = self.reverse(self.board[i])
            if not all(original_row == self.board[i]):
                board_changed = True
        return board_changed

    def move_up(self):
        """Move tiles up."""
        self.transpose()
        board_changed = self.move_left()
        self.transpose()
        return board_changed

    def move_down(self):
        """Move tiles down."""
        self.transpose()
        board_changed = self.move_right()
        self.transpose()
        return board_changed

    def _apply_move_without_spawn(self, action):
        """Apply a move without adding a new tile and return if the board changed."""
        if action == 0:
            return self.move_up()
        if action == 1:
            return self.move_right()
        if action == 2:
            return self.move_down()
        if action == 3:
            return self.move_left()
        raise ValueError(f"Unsupported action: {action}")

    def peek_move(self, action):
        """Return the resulting board and score delta if an action were applied."""
        original_board = self.board.copy()
        original_score = self.score

        board_changed = self._apply_move_without_spawn(action)
        score_delta = self.score - original_score
        preview_board = self.board.copy()

        self.board = original_board
        self.score = original_score

        return preview_board, score_delta, board_changed

    def get_available_actions(self):
        """Return actions that would change the board state."""
        original_board = self.board.copy()
        original_score = self.score

        available_actions = []
        for action in range(4):
            self.board = original_board.copy()
            self.score = original_score
            if self._apply_move_without_spawn(action):
                available_actions.append(action)

        self.board = original_board
        self.score = original_score

        return available_actions

    def step(self, action):
        """Apply an action to the game board (0: up, 1: right, 2: down, 3: left)."""
        board_changed = False
        if action == 0:
            board_changed = self.move_up()
        elif action == 1:
            board_changed = self.move_right()
        elif action == 2:
            board_changed = self.move_down()
        elif action == 3:
            board_changed = self.move_left()
        
        if board_changed:
            self.add_new_tile()
        
        game_over = self.is_game_over()
        
        return self.board, self.score, game_over


# Example usage
game = Game2048()
game_over = False
"""
while not game_over:
    action = np.random.choice([0, 1, 2, 3])  # Replace with your model's action
    _, score, game_over = game.step(action)
    print("Action:", action, "Score:", score)
"""