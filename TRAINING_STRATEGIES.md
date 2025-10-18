# GCN Training Strategies

This document describes the pluggable training strategy system for the GCN model.

## Overview

The GCN model now supports multiple training strategies that can be switched via configuration files. This allows you to:

1. **Supervised Learning** - Train using ground truth labels from optimal solver
2. **Reinforcement Learning** - Train directly using maximum load factor as reward

## Architecture

### Strategy Pattern

```
src/gcn/training/
├── __init__.py
├── base_strategy.py              # Abstract base class
├── supervised_strategy.py        # Supervised learning (default)
└── reinforcement_strategy.py     # RL with load factor reward
```

### Base Strategy Interface

All strategies inherit from `BaseTrainingStrategy` and implement:

- `compute_loss(model, batch_data, device)` - Compute loss/reward
- `backward_step(loss, optimizer, ...)` - Perform gradient update
- `get_metrics()` - Return training metrics
- `prepare_batch_data(batch, device)` - Prepare input tensors

## Available Strategies

### 1. Supervised Learning Strategy

**File**: [src/gcn/training/supervised_strategy.py](src/gcn/training/supervised_strategy.py)

**Description**: Traditional supervised learning using ground truth labels from Gurobi solver.

**How it works**:
1. Model predicts edge probabilities
2. Cross-entropy loss computed against optimal solution labels
3. Standard backpropagation updates parameters

**Loss function**: NLLLoss (Negative Log-Likelihood)

**Configuration**:
```json
{
  "training_strategy": "supervised"
}
```

**Metrics tracked**:
- `loss`: Cross-entropy loss
- `edge_error`: Percentage of incorrectly predicted edges

### 2. Reinforcement Learning Strategy

**File**: [src/gcn/training/reinforcement_strategy.py](src/gcn/training/reinforcement_strategy.py)

**Description**: Policy gradient learning using maximum load factor as reward signal.

**How it works**:
1. Model outputs edge probabilities (policy)
2. Beam search generates paths based on probabilities
3. Maximum load factor computed as reward
4. REINFORCE algorithm updates policy to maximize reward

**Algorithm**: REINFORCE (Policy Gradient) with baseline

**Reward design**:
- `load_factor`: reward = -max_load_factor (minimize load)
- `inverse_load_factor`: reward = 1/max_load_factor (maximize efficiency)
- Infeasible solutions: reward = -10.0 (large penalty)

**Configuration**:
```json
{
  "training_strategy": "reinforcement",
  "rl_reward_type": "load_factor",
  "rl_use_baseline": true,
  "rl_baseline_momentum": 0.9,
  "rl_entropy_weight": 0.01
}
```

**Parameters**:
- `rl_reward_type`: Type of reward (`"load_factor"` or `"inverse_load_factor"`)
- `rl_use_baseline`: Use moving average baseline for variance reduction
- `rl_baseline_momentum`: Momentum for baseline update (0.0-1.0)
- `rl_entropy_weight`: Weight for entropy regularization (encourages exploration)

**Metrics tracked**:
- `loss`: Policy gradient loss
- `mean_load_factor`: Average maximum load factor
- `reward`: Current reward value
- `advantage`: Reward - baseline
- `entropy`: Policy entropy (exploration measure)
- `baseline`: Current baseline value
- `is_feasible`: Whether solution is feasible (0 or 1)

## Usage

### Running with Supervised Learning (Default)

```bash
python scripts/gcn/train_gcn.py --config configs/gcn/default2.json
```

### Running with Reinforcement Learning

```bash
python scripts/gcn/train_gcn.py --config configs/gcn/rl_training.json
```

### Creating Custom Configurations

**Supervised example** (`configs/gcn/my_supervised.json`):
```json
{
  "expt_name": "my_experiment",
  "training_strategy": "supervised",
  "learning_rate": 0.001,
  "max_epochs": 100,
  ...
}
```

**Reinforcement example** (`configs/gcn/my_rl.json`):
```json
{
  "expt_name": "my_rl_experiment",
  "training_strategy": "reinforcement",
  "rl_reward_type": "load_factor",
  "rl_use_baseline": true,
  "rl_baseline_momentum": 0.9,
  "rl_entropy_weight": 0.01,
  "learning_rate": 0.0001,
  "max_epochs": 200,
  ...
}
```

## Key Differences

| Aspect | Supervised Learning | Reinforcement Learning |
|--------|-------------------|----------------------|
| **Requires labels** | Yes (from optimal solver) | No |
| **Loss function** | NLLLoss (cross-entropy) | Policy gradient |
| **Optimization target** | Match optimal solution | Minimize load factor |
| **Training signal** | Ground truth edges | Environment reward |
| **Exploration** | None | Entropy regularization |
| **Variance** | Low | Higher (reduced with baseline) |
| **Convergence** | Generally faster | May be slower |

## Implementation Details

### Model Changes

The GCN model's `forward()` method now supports optional loss computation:

```python
# With loss (backward compatible)
y_preds, loss = model.forward(x_edges, x_commodities, x_edges_capacity,
                              x_nodes, y_edges, edge_cw, compute_loss=True)

# Without loss (for RL strategy)
y_preds, _ = model.forward(x_edges, x_commodities, x_edges_capacity,
                           x_nodes, compute_loss=False)
```

### Trainer Integration

The `Trainer` class automatically selects and initializes the strategy:

```python
strategy_type = config.get('training_strategy', 'supervised')
if strategy_type == 'supervised':
    self.strategy = SupervisedLearningStrategy(config)
elif strategy_type == 'reinforcement':
    self.strategy = ReinforcementLearningStrategy(config)
```

## Adding New Strategies

To add a new training strategy:

1. Create a new file in `src/gcn/training/`
2. Inherit from `BaseTrainingStrategy`
3. Implement required methods:
   - `compute_loss()`
   - `backward_step()`
4. Add to `src/gcn/training/__init__.py`
5. Update `Trainer.__init__()` to recognize new strategy
6. Create example configuration file

**Example skeleton**:

```python
from .base_strategy import BaseTrainingStrategy

class MyCustomStrategy(BaseTrainingStrategy):
    def __init__(self, config):
        super().__init__(config)
        # Initialize strategy-specific parameters

    def compute_loss(self, model, batch_data, device=None):
        # Implement custom loss computation
        loss = ...
        metrics = {...}
        return loss, metrics

    def backward_step(self, loss, optimizer, accumulation_steps=1, batch_num=0):
        # Implement custom backward pass
        loss.backward()
        if (batch_num + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            return True
        return False
```

## Troubleshooting

### Issue: "Unknown training strategy: X"

**Solution**: Check that `training_strategy` in config is either `"supervised"` or `"reinforcement"`

### Issue: High variance in RL training

**Solutions**:
- Enable baseline: `"rl_use_baseline": true`
- Increase baseline momentum: `"rl_baseline_momentum": 0.95`
- Reduce learning rate: `"learning_rate": 0.0001`
- Increase batch size for more stable gradients

### Issue: RL training not improving

**Solutions**:
- Check reward design (try different `rl_reward_type`)
- Increase exploration: `"rl_entropy_weight": 0.05`
- Verify beam search is finding feasible solutions
- Reduce learning rate

### Issue: Supervised learning works but RL doesn't

**Diagnosis**: This is expected initially. RL requires more tuning:
- RL needs more episodes to converge
- Reward shaping may need adjustment
- Baseline helps reduce variance
- Start with smaller problems to validate

## References

- **REINFORCE algorithm**: Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
- **Policy Gradient methods**: Sutton & Barto, "Reinforcement Learning: An Introduction"
- **Beam Search for combinatorial optimization**: Original GCN paper implementation

## Future Improvements

Potential enhancements to the training strategy system:

1. **Actor-Critic methods**: Reduce variance further with value function baseline
2. **PPO (Proximal Policy Optimization)**: More stable RL training
3. **Hybrid strategies**: Combine supervised pre-training with RL fine-tuning
4. **Multi-objective rewards**: Consider both load factor and path length
5. **Curriculum learning**: Gradually increase problem difficulty
