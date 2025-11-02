"""
Type definitions for batch data structures.

This module defines TypedDict classes for batch data to ensure type safety
and provide clear documentation of expected tensor shapes and dtypes.
"""

from typing import TypedDict, Optional
import torch
from torch import Tensor


class RawBatchData(TypedDict):
    """
    Raw batch data from DatasetReader (DotDict format).
    All fields are numpy arrays.
    """
    edges: 'np.ndarray'  # [B, V, V] - Adjacency matrix (int)
    edges_capacity: 'np.ndarray'  # [B, V, V] - Edge capacity (float)
    nodes: 'np.ndarray'  # [B, V, C] - Node features (int: 0=none, 1=source, 2=target)
    commodities: 'np.ndarray'  # [B, C, 3] - Commodity data (int: source, target, demand)
    nodes_target: 'np.ndarray'  # [B, C, V] - Node flow targets (int)
    edges_target: 'np.ndarray'  # [B, V, V, C] - Edge flow targets (int)
    load_factor: 'np.ndarray'  # [B] - Optimal load factor (float)


class BatchData(TypedDict):
    """
    Processed batch data in PyTorch tensor format.

    Expected dtypes:
        - x_nodes: torch.long (for embedding layer)
        - x_commodities: torch.float
        - x_edges_capacity: torch.float
        - x_edges: torch.long (adjacency matrix)
        - edges_target: torch.long
        - nodes_target: torch.long
        - load_factor: torch.float
    """
    x_nodes: Tensor  # [B, V, C] - torch.long
    x_commodities: Tensor  # [B, C, 3] - torch.float
    x_edges_capacity: Tensor  # [B, V, V] - torch.float
    x_edges: Tensor  # [B, V, V] - torch.long
    edges_target: Tensor  # [B, V, V, C] - torch.long
    nodes_target: Tensor  # [B, C, V] - torch.long
    load_factor: Tensor  # [B] - torch.float


class EncoderInput(TypedDict):
    """
    Input tensors for GNN encoder.

    Expected dtypes:
        - x_nodes: torch.long (for embedding layer)
        - x_commodities: torch.float
        - x_edges_capacity: torch.float
        - x_edges_usage: torch.float (optional)
    """
    x_nodes: Tensor  # [B, V, C] - torch.long
    x_commodities: Tensor  # [B, C, 3] or [B, C] - torch.float
    x_edges_capacity: Tensor  # [B, V, V] - torch.float
    x_edges_usage: Optional[Tensor]  # [B, V, V] - torch.float


def validate_batch_types(batch_data: BatchData, strict: bool = True) -> None:
    """
    Validate that batch data has correct tensor dtypes.

    Args:
        batch_data: Batch data dictionary
        strict: If True, raises TypeError on mismatch. If False, prints warning.

    Raises:
        TypeError: If tensor dtypes don't match expected types (when strict=True)
    """
    expected_dtypes = {
        'x_nodes': torch.long,
        'x_commodities': torch.float32,
        'x_edges_capacity': torch.float32,
        'x_edges': torch.long,
        'edges_target': torch.long,
        'nodes_target': torch.long,
        'load_factor': torch.float32,
    }

    errors = []
    for key, expected_dtype in expected_dtypes.items():
        if key in batch_data:
            actual_dtype = batch_data[key].dtype
            if actual_dtype != expected_dtype:
                error_msg = f"Tensor '{key}' has dtype {actual_dtype}, expected {expected_dtype}"
                errors.append(error_msg)

    if errors:
        error_text = "\n".join(errors)
        if strict:
            raise TypeError(f"Batch data type validation failed:\n{error_text}")
        else:
            print(f"WARNING: Batch data type validation failed:\n{error_text}")


def validate_encoder_input_types(encoder_input: dict, strict: bool = True) -> None:
    """
    Validate that encoder input has correct tensor dtypes.

    Args:
        encoder_input: Encoder input dictionary (x_nodes, x_commodities, x_edges_capacity)
        strict: If True, raises TypeError on mismatch. If False, prints warning.

    Raises:
        TypeError: If tensor dtypes don't match expected types (when strict=True)
    """
    expected_dtypes = {
        'x_nodes': torch.long,
        'x_commodities': torch.float32,
        'x_edges_capacity': torch.float32,
        'x_edges_usage': torch.float32,
    }

    errors = []
    for key, expected_dtype in expected_dtypes.items():
        if key in encoder_input and encoder_input[key] is not None:
            actual_dtype = encoder_input[key].dtype
            if actual_dtype != expected_dtype:
                error_msg = f"Tensor '{key}' has dtype {actual_dtype}, expected {expected_dtype}"
                errors.append(error_msg)

    if errors:
        error_text = "\n".join(errors)
        if strict:
            raise TypeError(f"Encoder input type validation failed:\n{error_text}")
        else:
            print(f"WARNING: Encoder input type validation failed:\n{error_text}")
