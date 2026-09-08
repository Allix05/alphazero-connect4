"""Fixed-capacity FIFO buffer of self-play training examples."""
from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def add_game(self, examples: list[tuple[np.ndarray, np.ndarray, float]]):
        self.buffer.extend(examples)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        boards, policies, values = zip(*batch)
        return np.stack(boards), np.stack(policies), np.array(values, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)
