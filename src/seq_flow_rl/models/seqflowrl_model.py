"""
SeqFlowRLModel: Main Actor-Critic model for SeqFlowRL.

Integrates HybridGNNEncoder, PolicyHead, and ValueHead into a unified model.
"""

import torch
import torch.nn as nn

from .hybrid_gnn_encoder import HybridGNNEncoder
from .policy_head import PolicyHead
from .value_head import ValueHead


class SeqFlowRLModel(nn.Module):
    """
    SeqFlowRL Actor-Critic Model.

    Design decision #1: Shared encoder for Actor and Critic (confirmed)

    Architecture:
        Input (Graph State)
            ↓
        HybridGNNEncoder (shared)
            ↓
        ┌─────────────────┬─────────────────┐
        PolicyHead (Actor) ValueHead (Critic)
        ↓                  ↓
        Action Probs       State Value
    """

    def __init__(self, config, dtypeFloat=torch.float32, dtypeLong=torch.long):
        """
        Args:
            config: Configuration dictionary containing:
                - num_nodes: Number of nodes
                - num_commodities: Number of commodities
                - hidden_dim: Hidden dimension (default: 128)
                - num_layers: Number of GNN layers (default: 8)
                - aggregation: Aggregation method (default: 'mean')
                - dropout_rate: Dropout rate (default: 0.3)
                - action_type: Action space type ('node' or 'edge', default: 'node')
                - policy_head_mlp_layers: MLP layers for policy (default: 2)
                - value_head_mlp_layers: MLP layers for value (default: 3)
            dtypeFloat: Float data type (default: torch.float32)
            dtypeLong: Long data type (default: torch.long)
        """
        super().__init__()

        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        # Extract configuration
        self.num_nodes = config['num_nodes']
        self.num_commodities = config['num_commodities']
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 8)
        self.aggregation = config.get('aggregation', 'mean')
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.action_type = config.get('action_type', 'node')  # Default: node-level
        self.policy_head_mlp_layers = config.get('policy_head_mlp_layers', 2)
        self.value_head_mlp_layers = config.get('value_head_mlp_layers', 3)

        # Shared GNN Encoder (Decision #1: Shared)
        encoder_config = {
            'num_nodes': self.num_nodes,
            'num_commodities': self.num_commodities,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'aggregation': self.aggregation,
            'dropout_rate': self.dropout_rate,
        }
        self.encoder = HybridGNNEncoder(encoder_config)

        # Policy Head (Actor)
        self.actor = PolicyHead(
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            action_type=self.action_type,
            mlp_layers=self.policy_head_mlp_layers
        )

        # Value Head (Critic)
        self.critic = ValueHead(
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            num_commodities=self.num_commodities,
            mlp_layers=self.value_head_mlp_layers
        )

    def forward(self, x_nodes, x_commodities, x_edges_capacity, x_edges_usage=None,
                current_node=None, dst_node=None, commodity_idx=None,
                valid_edges_mask=None, mode='train'):
        """
        Forward pass through the SeqFlowRL model.

        Args:
            x_nodes: Node features [B, V, C]
            x_commodities: Commodity demand values [B, C] or [B, C, 3]
            x_edges_capacity: Edge capacity matrix [B, V, V]
            x_edges_usage: Edge usage matrix [B, V, V] (optional, for dynamic updates)
            current_node: Current node for action selection [B] or int (required for action)
            dst_node: Destination node [B] or int (required for action)
            commodity_idx: Commodity index [B] or int (required for action)
            valid_edges_mask: Mask for valid edges [B, V] (optional)
            mode: Mode of operation ('train', 'eval', 'encode_only')

        Returns:
            Depending on mode:
            - 'train' or 'eval': (action_probs, log_probs, entropy, state_value, node_features, edge_features)
            - 'encode_only': (node_features, edge_features, graph_embedding)
        """
        # Encode graph state with shared encoder
        node_features, edge_features, graph_embedding = self.encoder(
            x_nodes, x_commodities, x_edges_capacity, x_edges_usage
        )

        # If encode_only mode, return embeddings
        if mode == 'encode_only':
            return node_features, edge_features, graph_embedding

        # Actor: Compute action probabilities
        if current_node is not None and dst_node is not None and commodity_idx is not None:
            action_probs, log_probs, entropy = self.actor(
                node_features, edge_features,
                current_node, dst_node, commodity_idx,
                valid_edges_mask=valid_edges_mask
            )
        else:
            # No action requested, return None for actor outputs
            action_probs = None
            log_probs = None
            entropy = None

        # Critic: Compute state value
        state_value = self.critic(node_features)

        return action_probs, log_probs, entropy, state_value, node_features, edge_features

    def get_action_and_value(self, x_nodes, x_commodities, x_edges_capacity, x_edges_usage,
                            current_node, dst_node, commodity_idx, valid_edges_mask=None,
                            deterministic=False, temperature=1.0, top_p=0.9):
        """
        Sample action and compute value (for rollout).

        Args:
            x_nodes: Node features [B, V, C]
            x_commodities: Commodity demands [B, C] or [B, C, 3]
            x_edges_capacity: Edge capacity [B, V, V]
            x_edges_usage: Edge usage [B, V, V]
            current_node: Current node [B] or int
            dst_node: Destination node [B] or int
            commodity_idx: Commodity index [B] or int
            valid_edges_mask: Valid edges mask [B, V]
            deterministic: Use greedy action selection (default: False)
            temperature: Sampling temperature (default: 1.0)
            top_p: Nucleus sampling threshold (default: 0.9)

        Returns:
            action: Sampled action [B]
            log_prob: Log probability of action [B]
            entropy: Entropy of distribution [B]
            state_value: State value [B]
        """
        # Forward pass
        action_probs, log_probs, entropy, state_value, _, _ = self.forward(
            x_nodes, x_commodities, x_edges_capacity, x_edges_usage,
            current_node, dst_node, commodity_idx,
            valid_edges_mask=valid_edges_mask,
            mode='train'
        )

        # Sample action
        if deterministic:
            # Greedy: Select action with highest probability
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Stochastic: Sample from distribution
            action = self.actor.sample_action(action_probs, temperature=temperature, top_p=top_p)

        # Get log probability of selected action
        action_log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action, action_log_prob, entropy, state_value

    def evaluate_actions(self, x_nodes, x_commodities, x_edges_capacity, x_edges_usage,
                        current_node, dst_node, commodity_idx, actions, valid_edges_mask=None):
        """
        Evaluate actions (for PPO loss computation).

        Args:
            x_nodes: Node features [B, V, C]
            x_commodities: Commodity demands [B, C] or [B, C, 3]
            x_edges_capacity: Edge capacity [B, V, V]
            x_edges_usage: Edge usage [B, V, V]
            current_node: Current node [B] or int
            dst_node: Destination node [B] or int
            commodity_idx: Commodity index [B] or int
            actions: Actions to evaluate [B]
            valid_edges_mask: Valid edges mask [B, V]

        Returns:
            action_log_probs: Log probabilities of given actions [B]
            entropy: Entropy of distribution [B]
            state_value: State value [B]
        """
        # Forward pass
        action_probs, log_probs, entropy, state_value, _, _ = self.forward(
            x_nodes, x_commodities, x_edges_capacity, x_edges_usage,
            current_node, dst_node, commodity_idx,
            valid_edges_mask=valid_edges_mask,
            mode='train'
        )

        # Get log probabilities of given actions
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        return action_log_probs, entropy, state_value

    def load_pretrained_actor(self, supervised_model_path, device='cuda'):
        """
        Load pre-trained actor weights from supervised learning model.

        This implements the 2-phase training strategy (Decision #5: Option 5-A confirmed):
        - Phase 1: Supervised pre-training of Actor
        - Phase 2: RL fine-tuning of Actor + Critic

        Args:
            supervised_model_path: Path to supervised model checkpoint
            device: Device to load model on

        Returns:
            success: Whether loading was successful
        """
        try:
            checkpoint = torch.load(supervised_model_path, map_location=device)

            # Extract encoder and policy head weights
            # Note: This may require adapting keys depending on supervised model structure
            encoder_state_dict = {}
            actor_state_dict = {}

            for key, value in checkpoint['model_state_dict'].items():
                if 'encoder' in key or 'gcn' in key:
                    encoder_state_dict[key] = value
                elif 'policy' in key or 'mlp_edges' in key:
                    actor_state_dict[key] = value

            # Load encoder weights (with strict=False to allow missing/extra keys)
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
            self.actor.load_state_dict(actor_state_dict, strict=False)

            print(f"Successfully loaded pre-trained Actor from {supervised_model_path}")
            print(f"  - Encoder layers loaded: {len(encoder_state_dict)} parameters")
            print(f"  - Actor layers loaded: {len(actor_state_dict)} parameters")
            print(f"  - Critic initialized randomly (as per 2-phase training)")

            return True

        except Exception as e:
            print(f"Warning: Failed to load pre-trained model: {e}")
            print(f"Continuing with random initialization...")
            return False
