"""Connect Four game engine.

Board is a numpy int8 array of shape (ROWS, COLS). Cells hold +1 for the
player to move, -1 for the opponent, 0 for empty -- i.e. the board is
always stored from the *current player's* point of view (the "canonical"
form AlphaZero-style algorithms expect). This means after every move the
whole board is sign-flipped rather than tracking whose turn it is
separately.
"""
from __future__ import annotations

import numpy as np

ROWS = 6
COLS = 7
ACTION_SIZE = COLS
WIN_LENGTH = 4


class Connect4:
    """Immutable-ish helper around a canonical Connect Four board."""

    def __init__(self, board: np.ndarray | None = None):
        self.board = (
            np.zeros((ROWS, COLS), dtype=np.int8) if board is None else board
        )

    @staticmethod
    def initial_board() -> np.ndarray:
        return np.zeros((ROWS, COLS), dtype=np.int8)

    @staticmethod
    def valid_moves(board: np.ndarray) -> np.ndarray:
        """Boolean mask of length COLS: True where a disc can be dropped."""
        return board[0] == 0

    @staticmethod
    def drop(board: np.ndarray, col: int, player: int = 1) -> np.ndarray:
        """Return a new board with `player`'s disc dropped in `col`.

        Caller must ensure the move is legal.
        """
        new_board = board.copy()
        for row in range(ROWS - 1, -1, -1):
            if new_board[row, col] == 0:
                new_board[row, col] = player
                return new_board
        raise ValueError(f"Column {col} is full")

    @staticmethod
    def next_state(board: np.ndarray, action: int):
        """Apply the current player's move, return (canonical_next_board).

        The current player is always +1 in the input board; after the move
        we flip signs so the board is canonical for the *other* player.
        """
        played = Connect4.drop(board, action, player=1)
        return -played

    @staticmethod
    def check_winner(board: np.ndarray, player: int) -> bool:
        """True if `player` (+1/-1) has four in a row anywhere."""
        b = board == player
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if b[r, c : c + WIN_LENGTH].all():
                    return True
        # Vertical
        for c in range(COLS):
            for r in range(ROWS - WIN_LENGTH + 1):
                if b[r : r + WIN_LENGTH, c].all():
                    return True
        # Diagonal down-right
        for r in range(ROWS - WIN_LENGTH + 1):
            for c in range(COLS - WIN_LENGTH + 1):
                if all(b[r + i, c + i] for i in range(WIN_LENGTH)):
                    return True
        # Diagonal up-right
        for r in range(WIN_LENGTH - 1, ROWS):
            for c in range(COLS - WIN_LENGTH + 1):
                if all(b[r - i, c + i] for i in range(WIN_LENGTH)):
                    return True
        return False

    @staticmethod
    def is_full(board: np.ndarray) -> bool:
        return bool((board != 0).all())

    @staticmethod
    def game_ended(board: np.ndarray):
        """Return outcome from the perspective of the player who just moved:

        None  -> game not over
        1.0   -> the player who just moved (now -1 in canonical board) won
        -1.0  -> the mover to move next (+1) is already lost, i.e. opponent won
        1e-4  -> draw (small nonzero so it's distinguishable from "ongoing")
        """
        if Connect4.check_winner(board, -1):
            # The player who just played (before the sign flip) won.
            return 1.0
        if Connect4.check_winner(board, 1):
            return -1.0
        if Connect4.is_full(board):
            return 1e-4
        return None

    @staticmethod
    def encode(board: np.ndarray) -> np.ndarray:
        """Two-plane float32 tensor: [current player's pieces, opponent's]."""
        cur = (board == 1).astype(np.float32)
        opp = (board == -1).astype(np.float32)
        return np.stack([cur, opp], axis=0)

    @staticmethod
    def render(board: np.ndarray) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        lines = []
        for row in board:
            lines.append(" ".join(symbols[v] for v in row))
        lines.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(lines)
