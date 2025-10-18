"""
Training Strategies for GCN

This module provides different training strategies that can be plugged into the Trainer.
Each strategy defines how to compute loss and perform parameter updates.

Available strategies:
- SupervisedLearningStrategy: Traditional supervised learning with ground truth labels
- ReinforcementLearningStrategy: Policy gradient learning using max load factor as reward
"""

from .base_strategy import BaseTrainingStrategy
from .supervised_strategy import SupervisedLearningStrategy
from .reinforcement_strategy import ReinforcementLearningStrategy

__all__ = [
    'BaseTrainingStrategy',
    'SupervisedLearningStrategy',
    'ReinforcementLearningStrategy'
]
