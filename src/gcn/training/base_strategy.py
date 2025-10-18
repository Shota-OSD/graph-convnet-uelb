"""
Base Training Strategy Interface

This module defines the abstract base class for all training strategies.
Training strategies encapsulate different learning methods (supervised, reinforcement, etc.)
and can be plugged into the Trainer.
"""

from abc import ABC, abstractmethod
import torch


class BaseTrainingStrategy(ABC):
    """
    Abstract base class for training strategies.

    A training strategy defines how to:
    1. Compute loss/reward from model outputs
    2. Perform backward pass and parameter updates
    3. Track and report training metrics
    """

    def __init__(self, config):
        """
        Initialize training strategy.

        Args:
            config: Configuration dictionary/object
        """
        self.config = config
        self.metrics = {}

    @abstractmethod
    def compute_loss(self, model, batch_data, device=None):
        """
        Compute loss from model predictions and batch data.

        Args:
            model: The neural network model
            batch_data: Dictionary containing batch inputs and targets
            device: Device to run computations on

        Returns:
            loss: PyTorch tensor representing the loss
            metrics: Dictionary of additional metrics to log
        """
        pass

    @abstractmethod
    def backward_step(self, loss, optimizer, accumulation_steps=1, batch_num=0):
        """
        Perform backward pass and optimizer step.

        Args:
            loss: Loss tensor to backpropagate
            optimizer: PyTorch optimizer
            accumulation_steps: Number of steps to accumulate gradients
            batch_num: Current batch number (for accumulation logic)

        Returns:
            bool: True if optimizer step was performed, False otherwise
        """
        pass

    def get_metrics(self):
        """
        Get current training metrics.

        Returns:
            dict: Dictionary of metric name -> value
        """
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset metrics for new epoch."""
        self.metrics = {}

    def prepare_batch_data(self, batch, device):
        """
        Prepare batch data tensors and move to device.

        Args:
            batch: Raw batch from DataLoader
            device: Device to move tensors to

        Returns:
            dict: Dictionary containing prepared tensors
        """
        x_edges = torch.LongTensor(batch.edges).to(torch.long).contiguous().requires_grad_(False)
        x_edges_capacity = torch.FloatTensor(batch.edges_capacity).to(torch.float).contiguous().requires_grad_(False)
        x_nodes = torch.LongTensor(batch.nodes).to(torch.long).contiguous().requires_grad_(False)
        y_edges = torch.LongTensor(batch.edges_target).to(torch.long).contiguous().requires_grad_(False)
        batch_commodities = torch.LongTensor(batch.commodities).to(torch.long).contiguous().requires_grad_(False)
        x_commodities = batch_commodities[:, :, 2].to(torch.float)

        # Move to device
        x_edges = x_edges.to(device)
        x_edges_capacity = x_edges_capacity.to(device)
        x_nodes = x_nodes.to(device)
        y_edges = y_edges.to(device)
        batch_commodities = batch_commodities.to(device)
        x_commodities = x_commodities.to(device)

        return {
            'x_edges': x_edges,
            'x_edges_capacity': x_edges_capacity,
            'x_nodes': x_nodes,
            'y_edges': y_edges,
            'batch_commodities': batch_commodities,
            'x_commodities': x_commodities,
            'raw_batch': batch
        }
