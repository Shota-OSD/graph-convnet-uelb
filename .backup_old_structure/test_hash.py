#!/usr/bin/env python3
"""
新しいハッシュ計算方法をテストするスクリプト
"""

import json
import hashlib
from src.config.config_manager import ConfigManager

def generate_config_hash(config):
    """設定からハッシュ値を生成（新しい方法）"""
    # すべての設定を含むハッシュ計算
    config_for_hash = {}
    
    # 設定オブジェクトから辞書に変換（Settingsはdictを継承）
    config_dict = dict(config)
    
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
    
    for key in hash_keys:
        if key in config_dict:
            value = config_dict[key]
            # 数値、文字列、ブール値のみを含める
            if isinstance(value, (int, float, str, bool)) or value is None:
                config_for_hash[key] = value
    
    config_str = json.dumps(config_for_hash, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def test_different_configs():
    """異なる設定でハッシュ値が変わることをテスト"""
    
    config_files = [
        'configs/nsfnet_5_commodities.json',
        'configs/nsfnet_10_commodities.json',
        'configs/nsfnet_15_commodities.json',
        'configs/nsfnet_20_commodities.json',
        'configs/nsfnet_25_commodities.json',
    ]
    
    print("="*60)
    print("異なる設定ファイルのハッシュ値テスト")
    print("="*60)
    
    hash_values = {}
    
    for config_file in config_files:
        try:
            config_manager = ConfigManager(config_file)
            config = config_manager.get_config()
            hash_value = generate_config_hash(config)
            hash_values[config_file] = hash_value
            
            print(f"\n{config_file}:")
            print(f"  ハッシュ値: {hash_value}")
            print(f"  コモディティ数: {config.num_commodities}")
            print(f"  キャパシティ範囲: {config.capacity_lower}-{config.capacity_higher}")
            print(f"  需要範囲: {config.demand_lower}-{config.demand_higher}")
            
        except Exception as e:
            print(f"\n{config_file}: エラー - {e}")
    
    # 重複チェック
    unique_hashes = set(hash_values.values())
    print(f"\n" + "="*60)
    print(f"結果:")
    print(f"  設定ファイル数: {len(config_files)}")
    print(f"  ユニークなハッシュ値数: {len(unique_hashes)}")
    
    if len(unique_hashes) == len(config_files):
        print(f"  ✅ すべての設定ファイルが異なるハッシュ値を持っています")
    else:
        print(f"  ❌ 一部の設定ファイルが同じハッシュ値を持っています")
        print(f"  重複するハッシュ値:")
        hash_count = {}
        for hash_val in hash_values.values():
            hash_count[hash_val] = hash_count.get(hash_val, 0) + 1
        
        for hash_val, count in hash_count.items():
            if count > 1:
                print(f"    {hash_val}: {count}個の設定ファイル")
                for config_file, h in hash_values.items():
                    if h == hash_val:
                        print(f"      - {config_file}")

def test_rl_configs():
    """RL設定ファイルのハッシュ値テスト"""
    
    config_files = [
        'configs/nsfnet_5_commodities_rl.json',
        'configs/nsfnet_10_commodities_rl.json',
        'configs/nsfnet_15_commodities_rl.json',
        'configs/nsfnet_20_commodities_rl.json',
        'configs/nsfnet_25_commodities_rl.json',
    ]
    
    print("\n" + "="*60)
    print("RL設定ファイルのハッシュ値テスト")
    print("="*60)
    
    hash_values = {}
    
    for config_file in config_files:
        try:
            config_manager = ConfigManager(config_file)
            config = config_manager.get_config()
            
            # RL設定用のハッシュ計算
            config_for_hash = {}
            config_dict = dict(config)
            
            hash_keys = [
                # モデル構造関連
                'hidden_dims', 'learning_rate', 'gamma', 'epsilon', 'epsilon_decay',
                'n_action', 'obs_low', 'obs_high', 'K', 'max_step',
                
                # データ関連
                'num_commodities', 'capacity_lower', 'capacity_higher',
                'demand_lower', 'demand_higher', 'num_nodes', 'sample_size',
                
                # 学習関連
                'batch_size', 'max_epochs', 'episodes', 'test_episodes',
                
                # その他の重要な設定
                'solver_type', 'graph_model', 'expt_name'
            ]
            
            for key in hash_keys:
                if key in config_dict:
                    value = config_dict[key]
                    if isinstance(value, (int, float, str, bool, list)) or value is None:
                        config_for_hash[key] = value
            
            config_str = json.dumps(config_for_hash, sort_keys=True)
            hash_value = hashlib.md5(config_str.encode()).hexdigest()[:8]
            hash_values[config_file] = hash_value
            
            print(f"\n{config_file}:")
            print(f"  ハッシュ値: {hash_value}")
            print(f"  コモディティ数: {config.num_commodities}")
            print(f"  エピソード数: {config.episodes}")
            
        except Exception as e:
            print(f"\n{config_file}: エラー - {e}")
    
    # 重複チェック
    unique_hashes = set(hash_values.values())
    print(f"\n" + "="*60)
    print(f"RL設定結果:")
    print(f"  設定ファイル数: {len(config_files)}")
    print(f"  ユニークなハッシュ値数: {len(unique_hashes)}")
    
    if len(unique_hashes) == len(config_files):
        print(f"  ✅ すべてのRL設定ファイルが異なるハッシュ値を持っています")
    else:
        print(f"  ❌ 一部のRL設定ファイルが同じハッシュ値を持っています")

if __name__ == "__main__":
    test_different_configs()
    test_rl_configs()
