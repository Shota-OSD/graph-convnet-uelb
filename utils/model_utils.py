import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.beamsearch import *
from utils.graph_utils import *
from sklearn.utils.class_weight import compute_class_weight

def create_edge_class_weights(y_edges):
    # 1. Flatten the y_edges tensor to get edge labels
    edge_labels = y_edges.cpu().numpy().flatten()

    # 2. Compute class weights
    # Ensure unique classes are determined from edge_labels
    classes = np.unique(edge_labels)

    # Calculate class weights using compute_class_weight
    edge_cw = compute_class_weight("balanced", classes=classes, y=edge_labels)

    # Convert edge_cw to a PyTorch tensor if needed
    edge_cw_tensor = torch.tensor(edge_cw, dtype=torch.float32)

    return edge_cw_tensor


def loss_nodes(y_pred_nodes, y_nodes, node_cw):
    """
    Loss function for node predictions.

    Args:
        y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
        y_nodes: Targets for nodes (batch_size, num_nodes)
        node_cw: Class weights for nodes loss

    Returns:
        loss_nodes: Value of loss function
    
    """
     # Ensure tensors are contiguous
    y_pred_nodes = y_pred_nodes.contiguous()
    y_nodes = y_nodes.contiguous()

    # Node loss
    y = F.log_softmax(y_pred_nodes, dim=2)  # B x V x voc_nodes_out
    y = y.permute(0, 2, 1)  # B x voc_nodes x V
    criterion = nn.NLLLoss(weight=node_cw)
    loss_nodes = criterion(y, y_nodes)
    return loss_nodes


def loss_edges(y_pred_edges, y_edges, edge_cw):
    """
    Loss function for edge predictions.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, num_commodities)
        y_edges: Targets for edges (batch_size, num_nodes, num_nodes, num_commodities)
        edge_cw: Class weights for edges loss

    Returns:
        loss_edges: Value of loss function
    
    """
    # Ensure tensors are contiguous
    y_pred_edges = y_pred_edges.contiguous()
    y_edges = y_edges.contiguous().float()  # BCEWithLogitsLoss expects float type for targets

    # Edge loss (no need for log_softmax, directly use BCEWithLogitsLoss)
    # class weights are ignored for now
    criterion = nn.BCEWithLogitsLoss()

    
    loss_edges = criterion(y_pred_edges, y_edges)

    return loss_edges


def beamsearch_uelb_flows_nodes(y_pred_edges, commodities, edges_capacity, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='raw', random_start=False):
    """
    Performs beamsearch procedure on edge prediction matrices for UELB problems.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, num_commodities)
        commodities: Tensor containing source, target, and demand information (batch_size, num_commodities, 3)
        edges_capacity: Tensor containing edge capacities (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in the flow network
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns: Flow paths for commodities (batch_size, num_nodes)
    """
    
    if probs_type == 'raw':
        # Compute softmax over edge prediction matrix
        y = F.softmax(y_pred_edges, dim=3)  # B x V x V x num_commodities
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
    elif probs_type == 'logits':
        # Compute logits over edge prediction matrix
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x num_commodities
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number

    # Perform beamsearch
    beamsearch = Beamsearch(beam_size, batch_size, num_nodes, commodities, edges_capacity, dtypeFloat, dtypeLong, probs_type, random_start)
    trans_probs = y.gather(1, beamsearch.get_current_state())
    
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())
    
    ends = torch.zeros(batch_size, 1).type(dtypeLong)
    return beamsearch.get_hypothesis(ends)


def beamsearch_tour_nodes(y_pred_edges, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='raw', commotities_list):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible TSP tours.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, num_commodities)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in TSP tours
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns: UELB flows in terms of node ordering (batch_size, num_nodes, num_commodities)

    """
    target_state = commotities_list[:, :, 1] # (batch_size, num_commotities)
    
    if probs_type == 'raw':
        # Compute softmax over edge prediction matrix
        y = F.softmax(y_pred_edges, dim=3)  # B x V x V x num_commodities
    elif probs_type == 'logits':
        # Compute logits over edge prediction matrix
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x num_commodities
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number
    # Perform beamsearch
    beamsearch = Beamsearch(beam_size, batch_size, dtypeFloat, dtypeLong, probs_type)
    trans_probs = y.gather(1, beamsearch.get_current_state())
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        current_state = beamsearch.get_current_state()
        if current_state[:, 0, :] == target_state:
            break
        trans_probs = y.gather(1, current_state)
    # Find TSP tour with highest probability among beam_size candidates
    ends = torch.zeros(batch_size, 1).type(dtypeLong)
    return beamsearch.get_hypothesis(ends)


def beamsearch_tour_nodes_shortest(y_pred_edges, x_edges_values, beam_size, batch_size, num_nodes,
                                   dtypeFloat, dtypeLong, probs_type='raw', random_start=False):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible TSP tours.

    Final predicted tour is the one with the shortest tour length.
    (Standard beamsearch returns the one with the highest probability and does not take length into account.)

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, num_commodities)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in TSP tours
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        probs_type: Type of probability values being handled by beamsearch (either 'raw'/'logits'/'argmax'(TODO))
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns:
        shortest_tours: TSP tours in terms of node ordering (batch_size, num_nodes)

    """
    if probs_type == 'raw':
        # Compute softmax over edge prediction matrix
        y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
    elif probs_type == 'logits':
        # Compute logits over edge prediction matrix
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        # Consider the second dimension only
        y = y[:, :, :, 1]  # B x V x V
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number
    else:
        raise ValueError(f"Unsupported probs_type: {probs_type}")
    
    # Perform beamsearch
    beamsearch = Beamsearch(beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start)
    trans_probs = y.gather(1, beamsearch.get_current_state())
    
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())
    # Initially assign shortest_tours as most probable tours i.e. standard beamsearch
    ends = torch.zeros(batch_size, 1).type(dtypeLong)
    shortest_tours = beamsearch.get_hypothesis(ends)
    # Compute current tour lengths
    shortest_lens = [1e6] * len(shortest_tours)
    for idx in range(len(shortest_tours)):
        shortest_lens[idx] = tour_nodes_to_tour_len(shortest_tours[idx].cpu().numpy(),
                                                    x_edges_values[idx].cpu().numpy())
    # Iterate over all positions in beam (except position 0 --> highest probability)
    for pos in range(1, beam_size):
        ends = pos * torch.ones(batch_size, 1).type(dtypeLong)  # New positions
        hyp_tours = beamsearch.get_hypothesis(ends)
        for idx in range(len(hyp_tours)):
            hyp_nodes = hyp_tours[idx].cpu().numpy()
            hyp_len = tour_nodes_to_tour_len(hyp_nodes, x_edges_values[idx].cpu().numpy())
            # Replace tour in shortest_tours if new length is shorter than current best
            if hyp_len < shortest_lens[idx] and is_valid_tour(hyp_nodes, num_nodes):
                shortest_tours[idx] = hyp_tours[idx]
                shortest_lens[idx] = hyp_len
    return shortest_tours


def update_learning_rate(optimizer, lr):
    """
    Updates learning rate for given optimizer.

    Args:
        optimizer: Optimizer object
        lr: New learning rate

    Returns:
        optimizer: Updated optimizer object
        s
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def edge_error(y_pred, y_target, x_edges):
    """
    Computes edge error metrics for given batch prediction and targets.

    Args:
        y_pred: Edge predictions (batch_size, num_nodes, num_nodes, num_commodities)
        y_target: Edge targets (batch_size, num_nodes, num_nodes, num_commodities)
        x_edges: Adjacency matrix (batch_size, num_nodes, num_nodes)

    Returns:
        err_edges, err_flow, err_tsp, edge_err_idx, err_idx_flow, err_idx_uelb
    
    """
    # Make Binery output from y_pred
    y = (y_pred > 0.5).float()  # B x V x V x F

    # Edge error: Mask out edges which are not connected
    mask_no_edges = x_edges.unsqueeze(-1).long()
    err_edges, _ = _edge_error(y, y_target, mask_no_edges)

    # UELB flow edges error: Mask out edges which are not on true UELB flow
    mask_no_flow = y_target
    err_flow, err_idx_flow = _edge_error(y, y_target, mask_no_flow)
    # TSP tour edges + positively predicted edges error:
    # Mask out edges which are not on true TSP tours or are not predicted positively by model
    mask_no_uelb = ((y_target + y) > 0).long()
    err_uelb, err_idx_uelb = _edge_error(y, y_target, mask_no_uelb)

    return 100 * err_edges, 100 * err_flow, 100 * err_uelb, err_idx_flow, err_idx_uelb


def _edge_error(y, y_target, mask):
    """
    Helper method to compute edge errors.

    Args:
        y: Edge predictions (batch_size, num_nodes, num_nodes)
        y_target: Edge targets (batch_size, num_nodes, num_nodes)
        mask: Edges which are not counted in error computation (batch_size, num_nodes, num_nodes)

    Returns:
        err: Mean error over batch
        err_idx: One-hot array of shape (batch_size)- 1s correspond to indices which are not perfectly predicted
    
    """
    # Compute equalities between pred and target
    acc = (y == y_target).long()
    # Multipy by mask => set equality to 0 on disconnected edges
    acc = (acc * mask)
    #  Get accuracy of each y in the batch (sum of 1s in acc_edges divided by sum of 1s in edges mask)
    acc = acc.sum(dim=1).sum(dim=1).to(dtype=torch.float) / mask.sum(dim=1).sum(dim=1).to(dtype=torch.float)
    # Compute indices which are not perfect
    err_idx = (acc < 1.0)
    # Take mean over batch
    acc = acc.sum().to(dtype=torch.float).item() / acc.numel()
    # Compute error
    err = 1.0 - acc
    return err, err_idx
