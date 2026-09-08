const ROWS = C4.ROWS;
const COLS = C4.COLS;

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const newGameBtn = document.getElementById("newGame");
const difficultySel = document.getElementById("difficulty");
const humanFirstChk = document.getElementById("humanFirst");
const valueBar = document.getElementById("valueBar");
const valueNumber = document.getElementById("valueNumber");
const policyBarsEl = document.getElementById("policyBars");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");

const DIFFICULTY_SIMULATIONS = { easy: 20, medium: 120, hard: 400 };

let board, gameOver, humanTurn, busy, session;

function lowestEmptyRow(b, col) {
  for (let r = ROWS - 1; r >= 0; r--) {
    if (b[r][col] === 0) return r;
  }
  return -1;
}

function winningCellsFor(b, player) {
  const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
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
  policy.forEach((p) => {
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

// Runs one forward pass of the ONNX network. Board is from the perspective
// of whoever is about to move in it (matches the Python model.predict).
async function evaluateBoard(b) {
  const data = C4.encode(b);
  const tensor = new ort.Tensor("float32", data, [1, 2, ROWS, COLS]);
  const results = await session.run({ board: tensor });
  return { probs: results.policy.data, value: results.value.data[0] };
}

async function onColumnClick(col) {
  if (gameOver || busy || board[0][col] !== 0 || !humanTurn) return;

  const row = lowestEmptyRow(board, col);
  board[row][col] = 1;
  renderBoard();

  const winCells = winningCellsFor(board, 1);
  if (winCells) {
    gameOver = true;
    renderBoard(winCells);
    setStatus("You win!", "win");
    return;
  }
  if (C4.isFull(board)) {
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

  // The board is stored as 1=human/-1=AI internally is wrong for our MCTS,
  // which expects a canonical board where +1 = player to move (the AI here).
  const canonical = board.map((row) => row.map((v) => (v === 2 ? 1 : v === 1 ? -1 : 0)));

  const numSim = DIFFICULTY_SIMULATIONS[difficultySel.value] ?? DIFFICULTY_SIMULATIONS.medium;
  const policy = await AZMCTS.run(evaluateBoard, canonical, numSim);
  const column = policy.reduce((best, p, i) => (p > policy[best] ? i : best), 0);
  const { value } = await evaluateBoard(canonical);

  updateValueBar(value);
  updatePolicyBars(Array.from(policy));

  const row = lowestEmptyRow(board, column);
  board[row][column] = 2;

  const winCells = winningCellsFor(board, 2);
  busy = false;
  if (winCells) {
    gameOver = true;
    renderBoard(winCells);
    setStatus("AI wins.", "ai");
    return;
  }
  if (C4.isFull(board)) {
    gameOver = true;
    renderBoard();
    setStatus("Draw.", "");
    return;
  }

  humanTurn = true;
  renderBoard();
  setStatus("Your move.", "human");
}

async function newGame() {
  board = C4.initialBoard();
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

async function init() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
    loadingOverlay.classList.add("hidden");
    await newGame();
  } catch (err) {
    loadingText.textContent = "Failed to load the AI model: " + err.message;
  }
}

init();
