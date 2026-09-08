import numpy as np
import pytest

from alphazero.game import ACTION_SIZE, COLS, ROWS, Connect4


def test_initial_board_empty():
    board = Connect4.initial_board()
    assert board.shape == (ROWS, COLS)
    assert (board == 0).all()
    assert Connect4.valid_moves(board).all()


def test_drop_stacks_from_bottom():
    board = Connect4.initial_board()
    board = Connect4.drop(board, col=3, player=1)
    board = Connect4.drop(board, col=3, player=1)
    assert board[ROWS - 1, 3] == 1
    assert board[ROWS - 2, 3] == 1
    assert board[ROWS - 3, 3] == 0


def test_full_column_raises():
    board = Connect4.initial_board()
    for _ in range(ROWS):
        board = Connect4.drop(board, col=0, player=1)
    assert Connect4.valid_moves(board)[0] == False
    with pytest.raises(ValueError):
        Connect4.drop(board, col=0, player=1)


def test_next_state_flips_perspective():
    board = Connect4.initial_board()
    next_board = Connect4.next_state(board, action=0)
    # The disc just placed by the mover (+1) becomes -1 after the flip.
    assert next_board[ROWS - 1, 0] == -1


def test_horizontal_win_detected():
    board = Connect4.initial_board()
    for col in range(4):
        board = Connect4.drop(board, col=col, player=1)
    assert Connect4.check_winner(board, 1)
    assert not Connect4.check_winner(board, -1)


def test_vertical_win_detected():
    board = Connect4.initial_board()
    for _ in range(4):
        board = Connect4.drop(board, col=2, player=-1)
    assert Connect4.check_winner(board, -1)


def test_diagonal_win_detected():
    # check_winner is a pure array scan, so it's simplest (and unambiguous)
    # to place the diagonal directly rather than reproduce it via gravity.
    board = Connect4.initial_board()
    for i in range(4):
        board[2 + i, i] = 1  # down-right diagonal: (2,0),(3,1),(4,2),(5,3)
    assert Connect4.check_winner(board, 1)
    assert not Connect4.check_winner(board, -1)


def test_game_ended_none_when_ongoing():
    board = Connect4.initial_board()
    assert Connect4.game_ended(board) is None


def test_game_ended_returns_result_once_board_is_full():
    board = Connect4.initial_board()
    for col in range(COLS):
        for row in range(ROWS):
            board = Connect4.drop(board, col=col, player=1 if (row + col) % 2 == 0 else -1)
    assert Connect4.is_full(board)
    # Full board always yields a definite result: a win or the draw sentinel.
    assert Connect4.game_ended(board) is not None


def test_encode_shape_and_planes():
    board = Connect4.initial_board()
    board = Connect4.drop(board, col=0, player=1)
    board = Connect4.drop(board, col=1, player=-1)
    encoded = Connect4.encode(board)
    assert encoded.shape == (2, ROWS, COLS)
    assert encoded[0, ROWS - 1, 0] == 1.0  # current player's disc
    assert encoded[1, ROWS - 1, 1] == 1.0  # opponent's disc


def test_action_size_matches_cols():
    assert ACTION_SIZE == COLS
