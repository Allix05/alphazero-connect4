// Connect Four rules engine - canonical board (+1 = player to move, -1 =
// opponent, 0 = empty), mirroring alphazero/game.py so the JS demo makes
// exactly the same decisions as the Python training code.
const C4 = (() => {
  const ROWS = 6;
  const COLS = 7;
  const WIN_LENGTH = 4;

  function initialBoard() {
    return Array.from({ length: ROWS }, () => Array(COLS).fill(0));
  }

  function cloneBoard(board) {
    return board.map((row) => row.slice());
  }

  function validMoves(board) {
    const valid = new Array(COLS);
    for (let c = 0; c < COLS; c++) valid[c] = board[0][c] === 0;
    return valid;
  }

  function drop(board, col, player) {
    const next = cloneBoard(board);
    for (let r = ROWS - 1; r >= 0; r--) {
      if (next[r][col] === 0) {
        next[r][col] = player;
        return next;
      }
    }
    throw new Error(`Column ${col} is full`);
  }

  function nextState(board, action) {
    const played = drop(board, action, 1);
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) played[r][c] *= -1;
    }
    return played;
  }

  function checkWinner(board, player) {
    const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (board[r][c] !== player) continue;
        for (const [dr, dc] of dirs) {
          let ok = true;
          for (let k = 0; k < WIN_LENGTH; k++) {
            const rr = r + dr * k, cc = c + dc * k;
            if (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS || board[rr][cc] !== player) {
              ok = false;
              break;
            }
          }
          if (ok) return true;
        }
      }
    }
    return false;
  }

  function isFull(board) {
    return board[0].every((v) => v !== 0);
  }

  // Mirrors Connect4.game_ended: outcome from the perspective of the
  // player who just moved. null = ongoing, 1.0 = mover won, 1e-4 = draw.
  function gameEnded(board) {
    if (checkWinner(board, -1)) return 1.0;
    if (checkWinner(board, 1)) return -1.0;
    if (isFull(board)) return 1e-4;
    return null;
  }

  // Float32 [1,2,ROWS,COLS] tensor data (NCHW), current player's plane then opponent's.
  function encode(board) {
    const data = new Float32Array(2 * ROWS * COLS);
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const idx = r * COLS + c;
        if (board[r][c] === 1) data[idx] = 1;
        else if (board[r][c] === -1) data[ROWS * COLS + idx] = 1;
      }
    }
    return data;
  }

  return { ROWS, COLS, WIN_LENGTH, initialBoard, cloneBoard, validMoves, drop, nextState, checkWinner, isFull, gameEnded, encode };
})();
