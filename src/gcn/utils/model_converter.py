"""
Model Converter Utilities

Converts supervised learning models (voc_edges_out=2) to
reinforcement learning models (voc_edges_out=1) using logit difference method.
"""

import torch
from collections import OrderedDict


def convert_supervised_to_rl(supervised_state_dict, verbose=True):
    """
    Convert supervised model weights to RL model weights using logit difference.

    This preserves the relative confidence learned during supervised training:
    - Supervised: logit_diff = (W1-W0)*h + (b1-b0) where W1=class1, W0=class0
    - RL score: same logit_diff
    - Interpretation: score > 0 means P(use) > 0.5, score < 0 means P(use) < 0.5

    Args:
        supervised_state_dict: State dict from model with voc_edges_out=2
        verbose: Print conversion details

    Returns:
        rl_state_dict: State dict for model with voc_edges_out=1
    """
    rl_state_dict = OrderedDict()

    converted_layers = []

    for key, value in supervised_state_dict.items():
        # Remove DataParallel "module." prefix if present
        clean_key = key.replace('module.', '') if key.startswith('module.') else key

        # Check if this is the final MLP layer for edge classification
        # Support both old (fc_out) and new (output_layer) naming
        is_final_layer = ('mlp_edges' in clean_key and
                         ('fc_out' in clean_key or 'output_layer' in clean_key))

        if is_final_layer:
            if 'weight' in clean_key:
                # Shape: [2, hidden_dim] for supervised
                if value.shape[0] == 2:
                    # Compute W1 - W0 (class1 - class0)
                    new_weight = value[1:2, :] - value[0:1, :]  # [1, hidden_dim]
                    rl_state_dict[clean_key] = new_weight
                    converted_layers.append(f"{clean_key}: {value.shape} -> {new_weight.shape}")
                else:
                    # Not a binary classification layer, copy as-is
                    rl_state_dict[clean_key] = value

            elif 'bias' in clean_key:
                # Shape: [2] for supervised
                if value.shape[0] == 2:
                    # Compute b1 - b0
                    new_bias = value[1:2] - value[0:1]  # [1]
                    rl_state_dict[clean_key] = new_bias
                    converted_layers.append(f"{clean_key}: {value.shape} -> {new_bias.shape}")
                else:
                    # Not a binary classification layer, copy as-is
                    rl_state_dict[clean_key] = value
        else:
            # All other layers are compatible, copy as-is
            rl_state_dict[clean_key] = value

    if verbose:
        print("\n" + "="*70)
        print("MODEL CONVERSION: Supervised (voc_edges_out=2) -> RL (voc_edges_out=1)")
        print("="*70)
        print(f"Total parameters: {len(supervised_state_dict)}")
        print(f"Converted layers: {len(converted_layers)}")
        print("\nConverted layers using logit difference method:")
        for layer_info in converted_layers:
            print(f"  - {layer_info}")
        print("\nConversion method: score = (W1 - W0) * h + (b1 - b0)")
        print("  where W1/b1 = class 1 (edge used), W0/b0 = class 0 (edge not used)")
        print("="*70 + "\n")

    return rl_state_dict


def load_pretrained_supervised_model(rl_model, pretrained_path, device=None, verbose=True):
    """
    Load pre-trained supervised model weights into RL model.

    Args:
        rl_model: RL model instance with voc_edges_out=1
        pretrained_path: Path to saved supervised model (.pt file)
        device: Device to load model on
        verbose: Print loading details

    Returns:
        rl_model: Model with converted weights loaded
    """
    if verbose:
        print(f"\nLoading pre-trained supervised model from: {pretrained_path}")

    # Load supervised model checkpoint
    checkpoint = torch.load(pretrained_path, map_location=device)

    # Extract state dict (handle different checkpoint formats)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        supervised_state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        supervised_state_dict = checkpoint['state_dict']
    else:
        supervised_state_dict = checkpoint

    # Convert to RL format
    rl_state_dict = convert_supervised_to_rl(supervised_state_dict, verbose=verbose)

    # Load into RL model
    missing_keys, unexpected_keys = rl_model.load_state_dict(rl_state_dict, strict=False)

    if verbose:
        print("\nModel loading results:")
        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            print("  ✓ All weights loaded successfully!")

    return rl_model


def verify_conversion(supervised_model, rl_model, sample_input, device=None):
    """
    Verify that the conversion preserves the relative ordering of edge scores.

    Args:
        supervised_model: Original supervised model (voc_edges_out=2)
        rl_model: Converted RL model (voc_edges_out=1)
        sample_input: Sample input data for testing
        device: Device for computation

    Returns:
        dict: Verification results with correlation metrics
    """
    supervised_model.eval()
    rl_model.eval()

    with torch.no_grad():
        # Get supervised predictions
        supervised_output, _ = supervised_model.forward(
            *sample_input, compute_loss=False
        )
        # supervised_output shape: [B, V, V, C, 2]

        # Compute supervised scores as logit difference
        supervised_scores = supervised_output[:, :, :, :, 1] - supervised_output[:, :, :, :, 0]

        # Get RL predictions
        rl_output, _ = rl_model.forward(
            *sample_input, compute_loss=False
        )
        # rl_output shape: [B, V, V, C]

        # Flatten for comparison
        supervised_flat = supervised_scores.flatten().cpu()
        rl_flat = rl_output.flatten().cpu()

        # Compute correlation
        correlation = torch.corrcoef(torch.stack([supervised_flat, rl_flat]))[0, 1]

        # Compute mean absolute difference
        mae = torch.abs(supervised_flat - rl_flat).mean()

        # Check ranking consistency (Spearman-like)
        supervised_ranks = supervised_flat.argsort().argsort()
        rl_ranks = rl_flat.argsort().argsort()
        rank_correlation = torch.corrcoef(torch.stack([
            supervised_ranks.float(), rl_ranks.float()
        ]))[0, 1]

    results = {
        'correlation': correlation.item(),
        'mae': mae.item(),
        'rank_correlation': rank_correlation.item()
    }

    print("\n" + "="*70)
    print("CONVERSION VERIFICATION")
    print("="*70)
    print(f"Score correlation:    {results['correlation']:.4f} (should be ~1.0)")
    print(f"Mean absolute error:  {results['mae']:.4f} (should be ~0.0)")
    print(f"Rank correlation:     {results['rank_correlation']:.4f} (should be ~1.0)")

    if results['correlation'] > 0.99 and results['rank_correlation'] > 0.99:
        print("\n✓ Conversion verification PASSED!")
    else:
        print("\n✗ Conversion verification FAILED - scores do not match!")
    print("="*70 + "\n")

    return results
