"""
SeqFlowRL: Sequential Flow Reinforcement Learning
Hybrid approach combining GCN and Teal methods for network routing optimization.
"""

__version__ = "0.1.0"

# Main components
from .models import SeqFlowRLModel, HybridGNNEncoder, PolicyHead, ValueHead
from .algorithms import SequentialRolloutEngine
from .training import A2CStrategy, SeqFlowRLTrainer
from .utils import MaskGenerator

__all__ = [
    # Models
    "SeqFlowRLModel",
    "HybridGNNEncoder",
    "PolicyHead",
    "ValueHead",
    # Algorithms
    "SequentialRolloutEngine",
    # Training
    "A2CStrategy",
    "SeqFlowRLTrainer",
    # Utils
    "MaskGenerator",
]
