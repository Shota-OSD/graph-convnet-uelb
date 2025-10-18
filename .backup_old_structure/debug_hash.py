#!/usr/bin/env python3
"""
ハッシュ計算のデバッグ用スクリプト
"""

import json
import hashlib
from src.config.config_manager import ConfigManager

def debug_config_hash(config_file):
    """設定ファイルのハッシュ計算をデバッグ"""
    print(f"\n{'='*60}")
    print(f"デバッグ: {config_file}")
    print(f"{'='*60}")
    
    config_manager = ConfigManager(config_file)
    config = config_manager.get_config()
    
    # 設定オブジェクトから辞書に変換（Settingsはdictを継承）
    config_dict = dict(config)
    
    print(f"設定オブジェクトの型: {type(config)}")
    print(f"設定辞書のキー数: {len(config_dict)}")
    
    # ハッシュ計算に使用する設定を選択
    hash_keys = [
        # モデル構造関連
        'hidden_dim', 'num_layers', 'mlp_layers', 'node_dim',
        'voc_nodes_in', 'voc_nodes_out', 'voc_edges_in', 'voc_edges_out',
        'aggregation', 'dropout_rate', 'beam_size',
        
        # データ関連
        'num_commodities', 'capacity_lower', 'capacity_higher',
        'demand_lower', 'demand_higher', 'num_nodes', 'sample_size',
        
        # 学習関連
        'learning_rate', 'batch_size', 'max_epochs', 'decay_rate',
        
        # その他の重要な設定
        'solver_type', 'graph_model', 'expt_name'
    ]
    
    config_for_hash = {}
    
    print(f"\nハッシュ計算に使用される設定:")
    for key in hash_keys:
        if key in config_dict:
            value = config_dict[key]
            # 数値、文字列、ブール値のみを含める
            if isinstance(value, (int, float, str, bool)) or value is None:
                config_for_hash[key] = value
                print(f"  ✅ {key}: {value}")
            else:
                print(f"  ❌ {key}: {value} (型: {type(value)})")
        else:
            print(f"  ❌ {key}: 設定に存在しません")
    
    print(f"\nハッシュ計算用の設定辞書:")
    print(json.dumps(config_for_hash, sort_keys=True, indent=2))
    
    config_str = json.dumps(config_for_hash, sort_keys=True)
    hash_value = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    print(f"\nハッシュ値: {hash_value}")
    print(f"ハッシュ計算用のJSON文字列:")
    print(config_str)

def main():
    config_files = [
        'configs/nsfnet_5_commodities.json',
        'configs/nsfnet_10_commodities.json',
    ]
    
    for config_file in config_files:
        debug_config_hash(config_file)

if __name__ == "__main__":
    main()
