const ROWS = 6;
const COLS = 7;

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const newGameBtn = document.getElementById("newGame");
const difficultySel = document.getElementById("difficulty");
const humanFirstChk = document.getElementById("humanFirst");
const modelWarning = document.getElementById("modelWarning");
const valueBar = document.getElementById("valueBar");
const valueNumber = document.getElementById("valueNumber");
const policyBarsEl = document.getElementById("policyBars");

let board, gameOver, humanTurn, busy;

function emptyBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(0));
}

function checkWinner(b, player) {
  const dirs = [
    [0, 1], [1, 0], [1, 1], [1, -1],
  ];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (b[r][c] !== player) continue;
      for (const [dr, dc] of dirs) {
        const cells = [];
        let ok = true;
        for (let k = 0; k < 4; k++) {
          const rr = r + dr * k, cc = c + dc * k;
          if (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS || b[rr][cc] !== player) {
            ok = false;
            break;
          }
          cells.push([rr, cc]);
        }
        if (ok) return cells;
      }
    }
  }
  return null;
}

function isFull(b) {
  return b[0].every((v) => v !== 0);
}

function lowestEmptyRow(b, col) {
  for (let r = ROWS - 1; r >= 0; r--) {
    if (b[r][col] === 0) return r;
  }
  return -1;
}

function renderBoard(winningCells) {
  boardEl.innerHTML = "";
  const winSet = new Set((winningCells || []).map(([r, c]) => `${r},${c}`));

  for (let c = 0; c < COLS; c++) {
    const colEl = document.createElement("div");
    colEl.className = "column" + (gameOver || busy || board[0][c] !== 0 ? " disabled" : "");
    colEl.addEventListener("click", () => onColumnClick(c));

    for (let r = 0; r < ROWS; r++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      const val = board[r][c];
      if (val !== 0) {
        const disc = document.createElement("div");
        disc.className = "disc " + (val === 1 ? "human" : "ai");
        if (winSet.has(`${r},${c}`)) disc.classList.add("win");
        cell.appendChild(disc);
      }
      colEl.appendChild(cell);
    }
    boardEl.appendChild(colEl);
  }
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

function updateValueBar(value) {
  const half = Math.min(Math.abs(value), 1) * 50;
  const left = value >= 0 ? 50 : 50 - half;
  valueBar.style.left = left + "%";
  valueBar.style.width = half + "%";
  valueNumber.textContent = value.toFixed(2);
}

function updatePolicyBars(policy) {
  policyBarsEl.innerHTML = "";
  const max = Math.max(...policy, 0.001);
  policy.forEach((p, i) => {
    const col = document.createElement("div");
    col.className = "policy-col";
    const fill = document.createElement("div");
    fill.className = "policy-fill";
    fill.style.height = `${(p / max) * 100}%`;
    const pct = document.createElement("div");
    pct.className = "policy-pct";
    pct.textContent = `${Math.round(p * 100)}%`;
    col.appendChild(fill);
    col.appendChild(pct);
    policyBarsEl.appendChild(col);
  });
}

async function onColumnClick(col) {
  if (gameOver || busy || board[0][col] !== 0 || !humanTurn) return;

  const row = lowestEmptyRow(board, col);
  board[row][col] = 1;
  renderBoard();

  const winCells = checkWinner(board, 1);
  if (winCells) {
    gameOver = true;
    renderBoard(winCells);
    setStatus("You win!", "win");
    return;
  }
  if (isFull(board)) {
    gameOver = true;
    setStatus("Draw.", "");
    return;
  }

  humanTurn = false;
  await aiMove();
}

async function aiMove() {
  busy = true;
  setStatus("AI is thinking...", "ai");
  renderBoard();

  try {
    const res = await fetch("/api/ai-move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ board, difficulty: difficultySel.value }),
    });
    if (!res.ok) throw new Error(`server error ${res.status}`);
    const data = await res.json();

    modelWarning.classList.toggle("hidden", data.model_loaded);
    updateValueBar(data.value);
    updatePolicyBars(data.policy);

    const row = lowestEmptyRow(board, data.column);
    board[row][data.column] = 2;

    const winCells = checkWinner(board, 2);
    busy = false;
    if (winCells) {
      gameOver = true;
      renderBoard(winCells);
      setStatus("AI wins.", "ai");
      return;
    }
    if (isFull(board)) {
      gameOver = true;
      renderBoard();
      setStatus("Draw.", "");
      return;
    }

    humanTurn = true;
    renderBoard();
    setStatus("Your move.", "human");
  } catch (err) {
    busy = false;
    setStatus("Error contacting AI server: " + err.message, "");
    renderBoard();
  }
}

async function newGame() {
  board = emptyBoard();
  gameOver = false;
  busy = false;
  humanTurn = humanFirstChk.checked;
  updateValueBar(0);
  updatePolicyBars(Array(COLS).fill(1 / COLS));
  renderBoard();
  setStatus(humanTurn ? "Your move." : "AI is thinking...", humanTurn ? "human" : "ai");
  if (!humanTurn) {
    await aiMove();
  }
}

newGameBtn.addEventListener("click", newGame);

fetch("/api/health")
  .then((r) => r.json())
  .then((data) => modelWarning.classList.toggle("hidden", data.model_loaded))
  .catch(() => {});

newGame();
