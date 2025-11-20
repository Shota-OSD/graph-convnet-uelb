# Supervised Pre-training for SeqFlowRL

## Overview

This document describes the 2-phase training approach for SeqFlowRL:

- **Phase 1: Supervised Pre-training** - Learn from optimal solution paths via behavior cloning
- **Phase 2: RL Fine-tuning** - Improve beyond optimal solutions using A2C reinforcement learning

## Architecture

### Phase 1: Behavior Cloning

The policy network (Actor) learns to mimic expert actions from ground truth paths:

1. Extract sequential paths from `nodes_target` data (optimal solution)
2. For each step in the path, train the policy to predict the next node
3. Loss function: Cross-Entropy Loss between predicted and ground truth next nodes
4. Only the **Actor** (Policy Head + Encoder) is trained; **Critic** remains uninitialized

### Phase 2: RL Fine-tuning

Starting from pre-trained weights, the full model is trained with A2C:

1. Load pre-trained Actor weights
2. Initialize Critic randomly (or from scratch)
3. Train both Actor and Critic using A2C with reward signals
4. Use lower learning rate for fine-tuning (e.g., 0.0001 vs 0.001)

## Data Flow

### Input Data (from DatasetReader)

```python
batch.nodes_target: [B, C, V]  # Node visit order for each commodity
                               # Each element indicates the order a node is visited
                               # Example: [0, 2, 0, 1, 3, 0] means:
                               #   - Node 3 visited 1st (order=1)
                               #   - Node 1 visited 2nd (order=2)
                               #   - Node 4 visited 3rd (order=3)
```

### Extracted Paths

```python
path_sequences: [B, C, max_path_len]  # Sequential node IDs
                                      # Example: [3, 1, 4] (in visit order)
path_lengths: [B, C]                  # Length of each path
```

### Training Pairs

For each commodity, extract `(current_node, next_node)` pairs:

```
Path: [3, 1, 4, 7]
Pairs:
  - Step 0: current=3, next=1
  - Step 1: current=1, next=4
  - Step 2: current=4, next=7
```

## Usage

### Step 1: Supervised Pre-training

```bash
# Use pre-training config
python scripts/seq_flow_rl/train_seqflowrl.py --config configs/seqflowrl/seqflowrl_pretrain.json
```

**Key settings in `seqflowrl_pretrain.json`:**

```json
{
  "pretrain_mode": true,
  "max_epochs": 50,
  "learning_rate": 0.001,
  "checkpoint_dir": "saved_models/seqflowrl_pretrain/"
}
```

**Expected Output:**

```
Epoch 1/50 | Time: 12.34s | LR: 0.001000
  Train - Loss: 2.3456 | Accuracy: 45.67% | Predictions: 12800
  Val   - Loss: 2.1234 | Accuracy: 48.23% | Predictions: 1280

...

Epoch 50/50 | Time: 11.23s | LR: 0.001000
  Train - Loss: 0.4567 | Accuracy: 89.12% | Predictions: 12800
  Val   - Loss: 0.5123 | Accuracy: 86.45% | Predictions: 1280

→ Saving best model to saved_models/seqflowrl_pretrain/best_model.pt
```

### Step 2: RL Fine-tuning

Update `seqflowrl_mini.json`:

```json
{
  "pretrain_mode": false,
  "load_pretrained_model": true,
  "pretrained_model_path": "saved_models/seqflowrl_pretrain/best_model.pt",
  "learning_rate": 0.0001
}
```

Run RL training:

```bash
python scripts/seq_flow_rl/train_seqflowrl.py --config configs/seqflowrl/seqflowrl_mini.json
```

**Expected Output:**

```
======================================================================
LOADING PRE-TRAINED MODEL (2-PHASE TRAINING)
======================================================================
✓ Phase 1 (Supervised) → Phase 2 (RL Fine-tuning)
======================================================================

Epoch 1/100 | Time: 23.45s | LR: 0.000100
  Train - Loss: 1.2345 | Reward: 0.8765 | Load Factor: 0.6543 | Approx Ratio: 92.34%
  Val   - Load Factor: 0.6234 (min: 0.5123, max: 0.7456) | Approx Ratio: 94.56%
```

## Implementation Files

### Core Components

1. **Path Extraction Utility**
   - File: `src/seq_flow_rl/data/path_extraction.py`
   - Functions:
     - `extract_paths_from_node_target()` - Extract paths from node_target
     - `extract_step_pairs()` - Extract (current, next) pairs
     - `convert_batch_to_seqflowrl_format()` - Convert DatasetReader batch format

2. **Supervised Pre-training Strategy**
   - File: `src/seq_flow_rl/training/supervised_pretrain_strategy.py`
   - Class: `SupervisedPretrainStrategy`
   - Methods:
     - `train_step()` - Behavior cloning training step
     - `eval_step()` - Validation step

3. **Trainer Updates**
   - File: `src/seq_flow_rl/training/trainer.py`
   - Supports both `pretrain_mode=true` (Phase 1) and `pretrain_mode=false` (Phase 2)
   - Automatically switches between supervised and RL strategies

### Configuration Files

1. **Pre-training Config**: `configs/seqflowrl/seqflowrl_pretrain.json`
2. **RL Fine-tuning Config**: `configs/seqflowrl/seqflowrl_mini.json`

## Metrics

### Phase 1 Metrics

- **Loss**: Cross-entropy loss (lower is better)
- **Accuracy**: Percentage of correct next-node predictions (higher is better)
- **Total Predictions**: Number of (current, next) pairs processed

### Phase 2 Metrics

- **Loss**: A2C loss (actor + critic + entropy)
- **Reward**: Average reward per episode
- **Load Factor**: Network utilization (lower is better)
- **Approximation Ratio**: Model performance vs optimal solution (closer to 100% is better)
- **Completion Rate**: Percentage of commodities reaching destination

## Expected Results

### Phase 1 (Supervised)

- **Accuracy**: 80-95% after 50 epochs
- **Behavior**: Model learns to follow optimal paths almost perfectly

### Phase 2 (RL Fine-tuning)

- **Initial Performance**: Close to optimal (due to pre-training)
- **Final Performance**: May exceed optimal solution on unseen instances
- **Convergence**: Faster than training from scratch (fewer epochs needed)

## Benefits of 2-Phase Training

1. **Faster Convergence**: Pre-trained model starts from good initialization
2. **Better Exploration**: RL can explore beyond optimal solution
3. **Higher Completion Rate**: Pre-training ensures model learns valid paths
4. **Stable Training**: Less likely to get stuck in poor local minima

## Troubleshooting

### Low Accuracy in Phase 1

- Increase `max_epochs` (try 100)
- Increase `learning_rate` (try 0.002)
- Check data quality (ensure `nodes_target` contains valid paths)

### Phase 2 Doesn't Improve

- Verify pre-trained model is loaded correctly (check console output)
- Lower learning rate for fine-tuning (try 0.00005)
- Adjust reward function weights

### Model Doesn't Reach Destinations

- Check `max_path_length` (increase if needed)
- Review mask generation (ensure valid actions are not over-constrained)
- Increase pre-training epochs to improve initial policy
