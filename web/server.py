"""FastAPI backend serving the Connect Four web UI and AI move endpoint."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from alphazero.config import Config
from alphazero.game import ACTION_SIZE, COLS, ROWS, Connect4
from alphazero.mcts import MCTS
from alphazero.model import AlphaZeroNet

CHECKPOINT_PATH = os.environ.get("AZ_CHECKPOINT", os.path.join("checkpoints", "best.pt"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

DIFFICULTY_SIMULATIONS = {"easy": 15, "medium": 80, "hard": 300}

app = FastAPI(title="AlphaZero Connect Four")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cfg = Config()
_model = AlphaZeroNet(_cfg.channels, _cfg.num_res_blocks).to(_device)
_model_loaded = False
if os.path.exists(CHECKPOINT_PATH):
    _model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=_device))
    _model_loaded = True
_model.eval()


class MoveRequest(BaseModel):
    board: list[list[int]] = Field(..., description="6x7 grid: 0 empty, 1 human, 2 AI")
    difficulty: str = "medium"


class MoveResponse(BaseModel):
    column: int
    policy: list[float]
    value: float
    valid_moves: list[bool]
    model_loaded: bool


def _to_canonical(board: list[list[int]]) -> np.ndarray:
    arr = np.array(board, dtype=np.int8)
    if arr.shape != (ROWS, COLS):
        raise HTTPException(400, f"board must be {ROWS}x{COLS}")
    # It's the AI's turn, so AI pieces (2) become +1 and human pieces (1) become -1.
    canonical = np.zeros_like(arr)
    canonical[arr == 2] = 1
    canonical[arr == 1] = -1
    return canonical


@app.post("/api/ai-move", response_model=MoveResponse)
def ai_move(req: MoveRequest):
    board = _to_canonical(req.board)
    valid = Connect4.valid_moves(board)
    if not valid.any():
        raise HTTPException(400, "no valid moves left")
    if Connect4.game_ended(board) is not None:
        # Human's last move already ended the game before the AI gets to act.
        raise HTTPException(400, "game already over")

    num_sim = DIFFICULTY_SIMULATIONS.get(req.difficulty, DIFFICULTY_SIMULATIONS["medium"])
    mcts = MCTS(_model, _device, num_simulations=num_sim)
    policy = mcts.run(board, add_noise=False)
    column = int(np.argmax(policy))

    _, value = _model.predict(torch.from_numpy(Connect4.encode(board)), _device)

    return MoveResponse(
        column=column,
        policy=[float(p) for p in policy],
        value=float(value),
        valid_moves=[bool(v) for v in valid],
        model_loaded=_model_loaded,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _model_loaded, "device": str(_device)}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
