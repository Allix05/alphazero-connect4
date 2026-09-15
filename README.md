# AlphaZero Connect Four

[![CI](https://github.com/Allix05/alphazero-connect4/actions/workflows/ci.yml/badge.svg)](https://github.com/Allix05/alphazero-connect4/actions/workflows/ci.yml)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](LICENSE)

A from-scratch implementation of the **AlphaZero** algorithm (self-play reinforcement learning + Monte Carlo Tree Search guided by a neural network) applied to Connect Four, with a polished web UI to play against the trained agent and see its search visualized live.

No human game data. No hand-coded heuristics. The agent starts knowing nothing but the rules and gets stronger purely by playing itself, the same recipe DeepMind used for AlphaGo Zero / AlphaZero.

**[Play against it in your browser](https://allix05.github.io/alphazero-connect4/)** &mdash; the trained network runs client-side via ONNX Runtime Web, no backend required.

<!-- SCREENSHOT_PLACEHOLDER -->

## How it works

```
┌──────────────┐      self-play games       ┌──────────────────┐
│  Neural Net   │ ───────────────────────▶  │   Replay Buffer   │
│ (policy+value)│                            └──────────────────┘
│               │ ◀───────────────────────           │
└──────────────┘      guides MCTS search             │ sample batches
        ▲                                             ▼
        │                                    ┌──────────────────┐
        └──────────── trained on ──────────  │  Training step    │
                                              │ (policy + value   │
                                              │  loss, Adam)      │
                                              └──────────────────┘
                                                       │
                                              ┌──────────────────┐
                                              │  Arena gatekeep   │
                                              │ candidate vs best │
                                              └──────────────────┘
```

1. **[`alphazero/model.py`](alphazero/model.py)** — a residual CNN with two heads: a policy head (move probabilities over the 7 columns) and a value head (a `[-1, 1]` estimate of who's winning), the same dual-head design as AlphaZero's network.
2. **[`alphazero/mcts.py`](alphazero/mcts.py)** — PUCT Monte Carlo Tree Search: the network's policy biases which moves get explored, the network's value estimate replaces random rollouts, and visit counts after search become an improved policy target.
3. **[`alphazero/self_play.py`](alphazero/self_play.py)** — the network plays full games against itself using MCTS-improved move selection, producing `(board, search_policy, outcome)` training examples.
4. **[`alphazero/train.py`](alphazero/train.py)** — the main loop: generate self-play games → train a candidate network on the replay buffer → pit the candidate against the current best in an arena → promote it only if it wins convincingly. This gatekeeping keeps training monotonically improving instead of drifting.
5. **[`web/`](web/)** — a FastAPI backend that loads the trained checkpoint and a vanilla-JS frontend where you play against it and watch its policy/value output update after every move.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
pytest                      # run the test suite
```

### Play against the included checkpoint

```bash
uvicorn web.server:app --reload
# open http://127.0.0.1:8000
```

Or from the terminal:

```bash
python scripts/play_cli.py --checkpoint checkpoints/best.pt
```

### Train your own agent

```bash
# fast smoke test (~1-2 min on CPU, produces a weak but functional checkpoint)
python scripts/train.py --quick

# a real run - hours on CPU, much faster on GPU
python scripts/train.py --iterations 60 --games-per-iteration 100 --num-simulations 200
```

Each iteration: self-play games are generated with the current best network, a candidate is trained on the replay buffer, and the candidate only replaces the incumbent if it wins `≥55%` of an arena match against it (`alphazero/config.py` has every hyperparameter).

## Results

The checkpoint shipped in this repo (`checkpoints/best.pt`, also what's exported to the browser demo) comes from **15 self-play iterations** on a single CPU core (~35 minutes total: 24 self-play games + 80 MCTS simulations/move per iteration, arena-gatekept against the incumbent every time):

| | |
|---|---|
| Arena promotions | 5 / 15 iterations (candidate had to win &ge;55% of a 20-game arena match to replace the incumbent) |
| Policy loss | 1.95 &rarr; 1.55 |
| **Win rate vs. random-move baseline** | **39W / 1L / 0D over 40 games (98%)**, 100 MCTS simulations/move |

That's from well under an hour of training on a laptop CPU with no GPU. AlphaZero-style self-play keeps improving with more compute &mdash; a longer run (`python scripts/train.py --iterations 60 --games-per-iteration 100 --num-simulations 200`, ideally on a GPU) would push it considerably further toward optimal Connect Four play.

## Project structure

```
alphazero/         core library: game rules, MCTS, model, training loop
  game.py             Connect Four rules + canonical board encoding
  model.py            dual-head residual CNN
  mcts.py             PUCT search
  self_play.py        self-play game generation
  replay_buffer.py     FIFO training example buffer
  train.py            self-play -> train -> arena loop
  arena.py            match two agents against each other
  config.py           all hyperparameters in one place
scripts/
  train.py            CLI training entry point
  play_cli.py         play against a checkpoint in the terminal
  export_onnx.py      export a checkpoint to ONNX for the browser demo
web/
  server.py           FastAPI inference server
  static/             browser UI backed by the Python server
docs/                 fully static GitHub Pages demo (game rules, MCTS,
                      and ONNX Runtime Web inference all ported to JS --
                      no backend, this is what's live at the link above)
tests/                pytest suite for game rules + MCTS
.github/workflows/    CI: tests + a training smoke test on every push
```

## Two ways to play

1. **[Live browser demo](https://allix05.github.io/alphazero-connect4/)** (`docs/`) &mdash; the model is exported to ONNX and runs entirely client-side via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WebAssembly). The game rules and MCTS search are ported to plain JS (`docs/connect4.js`, `docs/mcts.js`) so it makes identical decisions to the Python implementation, verified against it numerically (see `scripts/export_onnx.py`). No server, no cold starts, free to host forever on GitHub Pages.
2. **Local FastAPI server** (`web/`) &mdash; runs the actual PyTorch model, useful when iterating on the network itself: `uvicorn web.server:app --reload`.

## Why these design choices

- **Canonical board encoding** (`+1` = player to move, `-1` = opponent) means the network only ever has to reason about "my pieces vs. their pieces," halving the input space it has to learn and letting the same network play both sides.
- **Arena gatekeeping** prevents a common failure mode in from-scratch self-play (a noisy training step silently making the agent worse) by only ever promoting checkpoints that demonstrably outperform the current best.
- **Dirichlet noise at the root** of self-play search (not during evaluation/play) forces early exploration without polluting the AI's actual playing strength when you play against it.

## License

All Rights Reserved — see [LICENSE](LICENSE). Source is public for portfolio/demonstration purposes; reuse requires permission.
