"""
Heuristic features for 2048 (for MLP state encoding).

All functions are NumPy-only and side-effect free.
"""

from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np


# ---------- basic helpers ----------
def log2_grid(board: np.ndarray) -> np.ndarray:
    """
    Return int log2 grid with empty->0.
    Example: 0->0, 2->1, 4->2, ..., 2048->11
    """
    b = board.copy()
    b[b == 0] = 1
    return np.log2(b).astype(np.int32)  # ints; empty->0

def _strip_zeros(seq: Sequence[int]) -> List[int]:
    return [x for x in seq if x != 0]


# ---------- heuristic features ----------
# Heuristic 1: check if max tile is sitting at a corner of the board
def corner_max_feature(board: np.ndarray) -> float:
    """1.0 if the global max tile sits in any corner; else 0.0."""
    maxv = int(board.max())
    corners = (board[0,0], board[0,3], board[3,0], board[3,3])
    return 1.0 if maxv in corners else 0.0

# Hueristic 2: top k elements forming a orthogonal chain
def topk_chain_ok(
    board: np.ndarray,
    k: int = 3,
    allow_equal: bool = True,     # allow v[i] == v[i-1]
    require_halving: bool = True, # allow v[i] == v[i-1] or v[i] == v[i-1]/2
) -> float:
    """
    Returns 1.0 if there exists a 4-connected (orthogonal) path covering the top-k
    tiles (with multiplicity, i.e., duplicates allowed) in descending order, where
    each step satisfies:
        - if require_halving=True: v[i] ∈ { v[i-1], v[i-1]/2 } (and allow_equal gates the first case)
        - else: v[i] <= v[i-1] (non-increasing)
    Turns are allowed. No diagonals. No cell reuse.
    """

    # 1) Top-k with multiplicity (no dedupe)
    flat = [v for v in board.flatten().tolist() if v > 0]
    if len(flat) < k:
        return 0.0
    flat.sort(reverse=True)               # multiset sorted desc
    target_vals = flat[:k]                # e.g., [512, 256, 256]

    # 2) Quick consistency check for the step rule
    def ok_step(prev, cur):
        if require_halving:
            if allow_equal and cur == prev:
                return True
            return (prev % 2 == 0) and (cur == prev // 2)
        else:
            return cur <= prev

    for i in range(1, k):
        if not ok_step(target_vals[i-1], target_vals[i]):
            return 0.0

    # 3) Map value -> list of positions
    pos = {}
    for v in set(target_vals):
        idx = np.where(board == v)
        cells = list(zip(idx[0].tolist(), idx[1].tolist()))
        if not cells:
            return 0.0
        pos[v] = cells

    # 4) DFS with adjacency + no cell reuse, matching sequence order
    from functools import lru_cache

    def neighbors(r, c):
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 4 and 0 <= nc < 4:
                yield nr, nc

    @lru_cache(None)
    def dfs(idx: int, r: int, c: int, used_mask: int) -> bool:
        if idx == k:
            return True
        want = target_vals[idx]
        for nr, nc in neighbors(r, c):
            bit = 1 << (nr * 4 + nc)
            if (used_mask & bit) != 0:
                continue
            if board[nr, nc] == want:
                if dfs(idx + 1, nr, nc, used_mask | bit):
                    return True
        return False

    # 5) Try all starts for the first value
    start_val = target_vals[0]
    for sr, sc in pos[start_val]:
        start_bit = 1 << (sr * 4 + sc)
        if dfs(1, sr, sc, start_bit):
            return 1.0
    return 0.0

# Heuristic 3: merge chains and potentials
def line_merge_potential(vals: Sequence[int]) -> int:
    """
    Count adjacent equal pairs after removing zeros (on a 1D line).
    Input values should be log2-coded (0==empty).
    """
    v = _strip_zeros(vals)
    return sum(1 for i in range(len(v) - 1) if v[i] == v[i + 1])

def row_col_merge_potential(logb: np.ndarray) -> Tuple[List[int], List[int]]:
    """Per-row and per-column merge potentials on a log2-coded grid."""
    rows = [line_merge_potential(logb[r, :].tolist()) for r in range(4)]
    cols = [line_merge_potential(logb[:, c].tolist()) for c in range(4)]
    return rows, cols

# Heuristic 4: monotonicity per line (after removing zeros)
def is_monotone(vals: Sequence[int]) -> int:
    """
    1 if the (zero-stripped) sequence is non-increasing OR non-decreasing; else 0.
    Input values should be log2-coded (0==empty).
    """
    v = _strip_zeros(vals)
    if len(v) <= 2:
        return 1
    non_inc = all(v[i] >= v[i + 1] for i in range(len(v) - 1))
    non_dec = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
    return 1 if (non_inc or non_dec) else 0

def row_col_monotonicity(logb: np.ndarray) -> Tuple[List[int], List[int]]:
    """Per-row and per-column monotonicity flags (0/1) on a log2-coded grid."""
    rows = [is_monotone(logb[r, :].tolist()) for r in range(4)]
    cols = [is_monotone(logb[:, c].tolist()) for c in range(4)]
    return rows, cols

# Heuristic 5: not to stuck into dead rows/cols, which causes only one forceful move.
def dead_lines_features(logb: np.ndarray) -> Tuple[float, float]:
    """
    Fraction of rows and columns with zero merge potential (log2-coded).
    Helps penalize 'frozen' lines that reduce future options.
    """
    r_merge, c_merge = row_col_merge_potential(logb)
    dead_r = sum(1 for x in r_merge if x == 0) / 4.0
    dead_c = sum(1 for x in c_merge if x == 0) / 4.0
    return float(dead_r), float(dead_c)

def legal_moves_norm(available_actions: Sequence[int]) -> float:
    """Normalized count of legal moves (0..1)."""
    return len(available_actions) / 4.0


# ---------- final encoder & (optional) shaping ----------
def encode_state_with_features(board: np.ndarray, available_actions: Sequence[int]) -> np.ndarray:
    """
    Build the MLP input vector:
      [ 16-D log2(board) , corner_max , legal_moves_norm ,
        row_merge(4) , col_merge(4) , row_mono(4) , col_mono(4) ,
        topk_chain_ok , dead_rows , dead_cols ]
    Returns float32 vector of length 37 by default.
    """
    logb = log2_grid(board)
    base16 = logb.astype(np.float32).ravel()

    corner = np.array([corner_max_feature(board)], dtype=np.float32)
    lm = np.array([legal_moves_norm(available_actions)], dtype=np.float32)

    rm, cm = row_col_merge_potential(logb)
    rmon, cmon = row_col_monotonicity(logb)
    feats = np.array(rm + cm + rmon + cmon, dtype=np.float32)

    topk = np.array([topk_chain_ok(board, k=3)], dtype=np.float32)
    deadr, deadc = dead_lines_features(logb)
    frozen = np.array([deadr, deadc], dtype=np.float32)

    return np.concatenate([base16, corner, lm, feats, topk, frozen], axis=0)

def shaping_reward(board: np.ndarray, available_actions: Sequence[int]) -> float:
    """
    Optional small reward shaping based on heuristics.
    Tune coefficients cautiously so score remains the main signal.
    """
    logb = log2_grid(board)
    corner = corner_max_feature(board)
    topk = topk_chain_ok(board, k=3)
    deadr, deadc = dead_lines_features(logb)

    # Coefficients are deliberately small.
    return 0.05 * corner + 0.025 * topk - 0.025 * (deadr + deadc)
