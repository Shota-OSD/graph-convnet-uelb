#!/usr/bin/env python3
"""
Deep Q-Network (DQN) model for RL-KSP
"""

import torch
import torch.nn as nn
from typing import List


class DQNModel(nn.Module):
    """
    Deep Q-Network model for reinforcement learning
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super(DQNModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 32, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
