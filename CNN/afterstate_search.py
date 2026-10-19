"""
2-ply expectimax move selection on top of the afterstate value network.

For each legal move a from board s:
    score(a) = r(s, a) + gamma * E_spawn[ max_a' ( r' + gamma * V(afterstate') ) ]
where the expectation runs over every empty cell of afterstate(s, a) receiving a 2 (p=0.9)
or a 4 (p=0.1). A spawned board with no legal move is worth 0. All second-level afterstates
are scored in one batched network call.

Boards are (4,4) uint8 exponents (0=empty, k=2^k), as produced by cnn_afterstate.board_to_exp.
Moves use a precomputed table over all 16^4 rows, so hundreds of boards slide at once.
"""

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3  # same action ids as game2048.Game2048.step
MAX_EXP = 15
SPAWNS = ((1, 0.9), (2, 0.1))  # (exponent, probability): tile 2 or tile 4


def _build_row_tables():
    """LEFT_ROWS[idx] = row after sliding left; ROW_GAIN[idx] = score gained. idx packs 4 exponents."""
    left = np.zeros((1 << 16, 4), dtype=np.uint8)
    gain = np.zeros(1 << 16, dtype=np.float32)
    for idx in range(1 << 16):
        tiles = [(idx >> (4 * k)) & 15 for k in range(4)]
        tiles = [e for e in tiles if e]
        out, score, i = [], 0, 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                out.append(min(tiles[i] + 1, MAX_EXP))
                score += 2 ** (tiles[i] + 1)
                i += 2
            else:
                out.append(tiles[i])
                i += 1
        left[idx, :len(out)] = out
        gain[idx] = score
    return left, gain


LEFT_ROWS, ROW_GAIN = _build_row_tables()


def _slide_left(boards: np.ndarray):
    r = boards.astype(np.int64)
    idx = r[..., 0] | (r[..., 1] << 4) | (r[..., 2] << 8) | (r[..., 3] << 12)  # (N,4)
    return LEFT_ROWS[idx], ROW_GAIN[idx].sum(axis=1)


def batch_move(boards: np.ndarray, action: int):
    """Slide every board in (N,4,4); return (afterstates, score gains (N,), changed mask (N,))."""
    if action == LEFT:
        out, gain = _slide_left(boards)
    elif action == RIGHT:
        out, gain = _slide_left(boards[:, :, ::-1])
        out = out[:, :, ::-1]
    elif action == UP:
        out, gain = _slide_left(boards.transpose(0, 2, 1))
        out = out.transpose(0, 2, 1)
    elif action == DOWN:
        out, gain = _slide_left(boards.transpose(0, 2, 1)[:, :, ::-1])
        out = out[:, :, ::-1].transpose(0, 2, 1)
    else:
        raise ValueError(f"Unsupported action: {action}")
    out = np.ascontiguousarray(out)
    return out, gain, (out != boards).any(axis=(1, 2))


def all_moves(boards: np.ndarray):
    """Return afterstates (N,4,4,4), gains (N,4), changed (N,4) for actions 0..3."""
    results = [batch_move(boards, a) for a in range(4)]
    return (np.stack([r[0] for r in results], axis=1),
            np.stack([r[1] for r in results], axis=1),
            np.stack([r[2] for r in results], axis=1))


def _spawned_children(afterstate: np.ndarray):
    """All boards reachable by one spawn on `afterstate`, with their probabilities."""
    flat = afterstate.reshape(-1)
    empty = np.flatnonzero(flat == 0)
    n = len(empty)
    boards = np.repeat(flat[None], n * len(SPAWNS), axis=0)
    probs = np.empty(n * len(SPAWNS), dtype=np.float32)
    for k, (exp, p) in enumerate(SPAWNS):
        rows = np.arange(k * n, (k + 1) * n)
        boards[rows, empty] = exp
        probs[rows] = p / n
    return boards.reshape(-1, 4, 4), probs


def expectimax_action(agent, board: np.ndarray, reward_scale: float) -> int | None:
    """Best action for `board` under 2-ply expectimax, or None when no move is legal."""
    gamma = agent.gamma
    first, first_gain, first_ok = (x[0] for x in all_moves(board[None]))
    legal = np.flatnonzero(first_ok)
    if legal.size == 0:
        return None

    children, probs, owner = [], [], []
    for a in legal:
        b, p = _spawned_children(first[a])
        children.append(b)
        probs.append(p)
        owner.append(np.full(len(p), a))
    children = np.concatenate(children)
    probs = np.concatenate(probs)
    owner = np.concatenate(owner)

    second, second_gain, second_ok = all_moves(children)  # (M,4,4,4), (M,4), (M,4)
    q = np.full(second_ok.shape, -np.inf, dtype=np.float32)
    if second_ok.any():
        q[second_ok] = reward_scale * second_gain[second_ok] + gamma * agent.values(second[second_ok])
    child_value = np.where(second_ok.any(axis=1), q.max(axis=1), 0.0)  # dead boards are worth 0

    expected = np.bincount(owner, weights=probs * child_value, minlength=4)
    scores = reward_scale * first_gain[legal] + gamma * expected[legal]
    return int(legal[np.argmax(scores)])
