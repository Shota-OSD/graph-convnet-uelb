"""
Supervised Learning Strategy

This strategy implements the traditional supervised learning approach
where the model learns to predict edge usage by minimizing the difference
between predictions and ground truth labels from an optimal solver.
"""

import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .base_strategy import BaseTrainingStrategy
from ..models.model_utils import loss_edges, edge_error


class SupervisedLearningStrategy(BaseTrainingStrategy):
    """
    Supervised learning strategy using ground truth labels.

    This is the original training method where:
    1. Model predicts edge probabilities
    2. Loss is computed against optimal solution labels (y_edges)
    3. Standard backpropagation updates model parameters

    Loss function: NLLLoss (Negative Log-Likelihood)
    """

    def __init__(self, config):
        super().__init__(config)
        self.edge_cw = None  # Class weights for balanced learning

    def compute_loss(self, model, batch_data, device=None):
        """
        Compute supervised loss using ground truth labels.

        Args:
            model: GCN model
            batch_data: Dictionary with 'x_edges', 'x_commodities', 'x_edges_capacity',
                       'x_nodes', 'y_edges', 'batch_commodities'
            device: Device for computation

        Returns:
            loss: Computed loss tensor
            metrics: Dictionary with 'edge_error'
        """
        x_edges = batch_data['x_edges']
        x_commodities = batch_data['x_commodities']
        x_edges_capacity = batch_data['x_edges_capacity']
        x_nodes = batch_data['x_nodes']
        y_edges = batch_data['y_edges']

        # Compute class weights if not already computed
        if self.edge_cw is None or type(self.edge_cw) != torch.Tensor:
            edge_labels = y_edges.cpu().numpy().flatten()
            self.edge_cw = compute_class_weight(
                "balanced",
                classes=np.unique(edge_labels),
                y=edge_labels
            )
            self.edge_cw = torch.tensor(self.edge_cw, dtype=torch.float32)
            if device:
                self.edge_cw = self.edge_cw.to(device)

        # Forward pass through model
        y_preds, loss = model.forward(
            x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges, self.edge_cw
        )

        # Compute edge error for monitoring
        err_edges, _ = edge_error(y_preds, y_edges, x_edges)

        # Store metrics
        metrics = {
            'edge_error': err_edges
        }

        return loss, metrics

    def backward_step(self, loss, optimizer, accumulation_steps=1, batch_num=0):
        """
        Perform standard backward pass with gradient accumulation.

        Args:
            loss: Loss tensor
            optimizer: PyTorch optimizer
            accumulation_steps: Number of batches to accumulate gradients
            batch_num: Current batch number

        Returns:
            bool: True if optimizer step was performed
        """
        # Scale loss by accumulation steps
        loss = loss.mean() / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps batches
        if (batch_num + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            return True

        return False

    def reset_metrics(self):
        """Reset metrics and class weights for new epoch."""
        super().reset_metrics()
        self.edge_cw = None
