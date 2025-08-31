#!/usr/bin/env python3
"""
ハッシュ値から設定を逆算するスクリプト
"""

import json
import hashlib
import itertools

def generate_config_hash(config):
    """設定からハッシュ値を生成"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def reverse_hash(target_hash, known_values=None):
    """
    ハッシュ値から設定を逆算
    
    Args:
        target_hash (str): 目標のハッシュ値（8文字）
        known_values (dict): 既知の設定値
    
    Returns:
        list: 一致する設定のリスト
    """
    
    # デフォルトの設定範囲
    default_ranges = {
        'hidden_dim': [32, 64, 128, 256],
        'num_layers': [5, 10, 15, 20],
        'mlp_layers': [1, 2, 3],
        'node_dim': [5, 10, 15, 20],
        'voc_nodes_in': [20, 30, 40, 50],
        'voc_nodes_out': [1, 2, 3],
        'voc_edges_in': [2, 3, 4],
        'voc_edges_out': [1, 2, 3],
        'aggregation': ['mean', 'sum', 'max'],
        'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    # 既知の値がある場合は固定
    if known_values:
        for key, value in known_values.items():
            if key in default_ranges:
                default_ranges[key] = [value]
    
    # 全組み合わせを生成
    keys = list(default_ranges.keys())
    values = list(default_ranges.values())
    
    matching_configs = []
    
    print(f"ハッシュ値 '{target_hash}' を逆算中...")
    print(f"探索範囲: {len(list(itertools.product(*values)))} 通り")
    
    for i, combination in enumerate(itertools.product(*values)):
        if i % 10000 == 0:
            print(f"進捗: {i} 通りチェック済み")
        
        config = dict(zip(keys, combination))
        hash_value = generate_config_hash(config)
        
        if hash_value == target_hash:
            matching_configs.append(config)
            print(f"一致する設定を発見: {config}")
    
    return matching_configs

def verify_hash(config, target_hash):
    """設定とハッシュ値が一致するか検証"""
    hash_value = generate_config_hash(config)
    return hash_value == target_hash

def main():
    # 目標のハッシュ値
    target_hash = "49a6a2e9"
    
    print("="*60)
    print("ハッシュ値から設定を逆算")
    print("="*60)
    
    # 方法1: 既知の設定で検証
    print("\n1. 既知の設定で検証")
    known_config = {
        'hidden_dim': 64,
        'num_layers': 10,
        'mlp_layers': 2,
        'node_dim': 10,
        'voc_nodes_in': 30,
        'voc_nodes_out': 2,
        'voc_edges_in': 3,
        'voc_edges_out': 2,
        'aggregation': 'mean',
        'dropout_rate': 0.5,
    }
    
    if verify_hash(known_config, target_hash):
        print(f"✅ 既知の設定がハッシュ値 '{target_hash}' と一致します")
        print("設定:")
        for key, value in known_config.items():
            print(f"  {key}: {value}")
    else:
        print(f"❌ 既知の設定がハッシュ値 '{target_hash}' と一致しません")
    
    # 方法2: 部分的な既知値で逆算
    print("\n2. 部分的な既知値で逆算")
    partial_known = {
        'hidden_dim': 64,
        'num_layers': 10,
        'mlp_layers': 2,
        'node_dim': 10,
        'voc_nodes_in': 30,
        'voc_nodes_out': 2,
        'voc_edges_in': 3,
        'voc_edges_out': 2,
        'aggregation': 'mean',
        # dropout_rate は未知
    }
    
    matching_configs = reverse_hash(target_hash, partial_known)
    
    if matching_configs:
        print(f"\n✅ {len(matching_configs)} 個の一致する設定を発見しました:")
        for i, config in enumerate(matching_configs, 1):
            print(f"\n設定 {i}:")
            for key, value in config.items():
                print(f"  {key}: {value}")
    else:
        print(f"\n❌ 一致する設定が見つかりませんでした")
    
    # 方法3: 完全な逆算（時間がかかる）
    print("\n3. 完全な逆算（時間がかかります）")
    print("この処理は非常に時間がかかるため、コメントアウトしています")
    print("実行する場合は以下の行のコメントを外してください:")
    print("# matching_configs = reverse_hash(target_hash)")

if __name__ == "__main__":
    main()
