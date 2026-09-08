"""Central hyperparameter configuration for training and search."""
from dataclasses import dataclass


@dataclass
class Config:
    # Network
    channels: int = 64
    num_res_blocks: int = 6

    # MCTS
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25

    # Self-play
    games_per_iteration: int = 40
    temperature_moves: int = 12  # play stochastically for first N plies, then greedy
    max_replay_buffer: int = 60_000

    # Training
    iterations: int = 20
    epochs_per_iteration: int = 4
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Arena (gatekeeping new checkpoints)
    arena_games: int = 20
    arena_win_rate_threshold: float = 0.55
    arena_num_simulations: int = 80

    checkpoint_dir: str = "checkpoints"
    seed: int = 42
