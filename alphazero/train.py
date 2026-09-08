"""Full AlphaZero training loop: self-play -> train -> arena gatekeeping."""
from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .arena import mcts_agent, play_match, random_agent
from .config import Config
from .model import AlphaZeroNet
from .replay_buffer import ReplayBuffer
from .self_play import play_one_game


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_step(model, optimizer, boards, policies, values, device):
    model.train()
    boards_t = torch.from_numpy(boards).to(device)
    policies_t = torch.from_numpy(policies).to(device)
    values_t = torch.from_numpy(values).to(device)

    logits, value_pred = model(boards_t)
    policy_loss = -(policies_t * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(value_pred, values_t)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(policy_loss.item()), float(value_loss.item())


def run_training(cfg: Config, log=print):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = get_device()
    log(f"Using device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = AlphaZeroNet(cfg.channels, cfg.num_res_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.max_replay_buffer)

    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        log(f"Resumed from {best_path}")
    else:
        torch.save(model.state_dict(), best_path)

    history = []

    for iteration in range(1, cfg.iterations + 1):
        t0 = time.time()

        # --- Self-play ---
        for _ in range(cfg.games_per_iteration):
            examples = play_one_game(model, device, cfg)
            buffer.add_game(examples)
        t_selfplay = time.time() - t0

        # --- Train a candidate network ---
        candidate = copy.deepcopy(model)
        cand_optimizer = torch.optim.Adam(candidate.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        cand_optimizer.load_state_dict(optimizer.state_dict())

        policy_losses, value_losses = [], []
        steps_per_epoch = max(1, len(buffer) // cfg.batch_size)
        for _ in range(cfg.epochs_per_iteration):
            for _ in range(steps_per_epoch):
                boards, policies, values = buffer.sample(cfg.batch_size)
                pl, vl = train_step(candidate, cand_optimizer, boards, policies, values, device)
                policy_losses.append(pl)
                value_losses.append(vl)
        t_train = time.time() - t0 - t_selfplay

        # --- Arena gatekeeping: only promote candidate if it beats the incumbent ---
        cand_wins, best_wins, draws = play_match(
            mcts_agent(candidate, device, cfg.arena_num_simulations),
            mcts_agent(model, device, cfg.arena_num_simulations),
            num_games=cfg.arena_games,
        )
        decisive = cand_wins + best_wins
        win_rate = cand_wins / decisive if decisive > 0 else 0.5
        promoted = win_rate >= cfg.arena_win_rate_threshold
        if promoted:
            model = candidate
            optimizer = cand_optimizer
            torch.save(model.state_dict(), best_path)

        t_arena = time.time() - t0 - t_selfplay - t_train

        row = {
            "iteration": iteration,
            "buffer_size": len(buffer),
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
            "value_loss": float(np.mean(value_losses)) if value_losses else None,
            "arena_win_rate": win_rate,
            "promoted": promoted,
            "t_selfplay_s": round(t_selfplay, 1),
            "t_train_s": round(t_train, 1),
            "t_arena_s": round(t_arena, 1),
        }
        history.append(row)
        log(f"[iter {iteration}/{cfg.iterations}] "
            f"buffer={row['buffer_size']} "
            f"policy_loss={row['policy_loss']:.3f} value_loss={row['value_loss']:.3f} "
            f"arena_win_rate={win_rate:.2f} {'PROMOTED' if promoted else 'kept incumbent'} "
            f"(selfplay {t_selfplay:.1f}s / train {t_train:.1f}s / arena {t_arena:.1f}s)")

        torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, f"iter_{iteration}.pt"))

    torch.save(model.state_dict(), best_path)
    return model, history


def evaluate_vs_random(model, device, num_simulations: int = 100, num_games: int = 40):
    wins, losses, draws = play_match(
        mcts_agent(model, device, num_simulations),
        random_agent(),
        num_games=num_games,
    )
    return wins, losses, draws
