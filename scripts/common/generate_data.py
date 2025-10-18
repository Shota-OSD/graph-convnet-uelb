#!/usr/bin/env python3
"""
データ生成専用スクリプト
main.pyから独立してデータセットを生成する
"""

import sys
import argparse
import os
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common.config.config_manager import ConfigManager
from src.common.data_management.create_data_files import create_data_files

def main():
    """データ生成のメイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Dataset Generator for UELB')
    parser.add_argument('--config', type=str, default='configs/default2.json',
                       help='設定ファイルのパス (default: configs/default2.json)')
    parser.add_argument('--modes', type=str, nargs='+', default=['train', 'val', 'test'],
                       choices=['train', 'val', 'test'],
                       help='生成するデータモード (default: train val test)')
    parser.add_argument('--force', action='store_true',
                       help='確認なしで既存データを削除して再生成')
    parser.add_argument('--clean-only', action='store_true',
                       help='データを削除するのみ（再生成しない）')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DATASET GENERATION")
    print("="*60)
    print(f"Config file: {args.config}")
    print(f"Modes to generate: {', '.join(args.modes)}")
    
    # 設定の初期化
    try:
        config_manager = ConfigManager(args.config)
        config = config_manager.get_config()
        print(f"Number of commodities: {config.num_commodities}")
        print(f"Number of nodes: {config.num_nodes}")
        print(f"Train data: {config.num_train_data}")
        print(f"Val data: {config.num_val_data}")
        print(f"Test data: {config.num_test_data}")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # 既存データの確認と削除
    existing_dirs = []
    for mode in args.modes:
        data_dir = f"./data/{mode}_data"
        if os.path.exists(data_dir):
            existing_dirs.append(data_dir)
    
    if existing_dirs:
        print(f"\n既存のデータディレクトリが見つかりました:")
        for dir_path in existing_dirs:
            print(f"  - {dir_path}")
        
        if not args.force:
            ans = input("\n既存データを削除して再生成しますか？ (y/n): ").strip().lower()
            if ans not in ["y", "yes"]:
                print("データ生成をキャンセルしました。")
                sys.exit(0)
    
    # 既存データの削除
    for mode in args.modes:
        data_dir = f"./data/{mode}_data"
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"削除: {data_dir}")
    
    if args.clean_only:
        print("\nデータ削除が完了しました。")
        return
    
    # 新しいデータの生成
    print(f"\n新しいデータを生成しています...")
    for mode in args.modes:
        print(f"\n{mode.upper()} データを生成中...")
        try:
            create_data_files(config, data_mode=mode)
            print(f"✓ {mode} データの生成が完了しました")
        except Exception as e:
            print(f"✗ {mode} データの生成に失敗しました: {e}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("データ生成が完了しました！")
    print("="*60)
    
    # 生成されたデータの確認
    print("\n生成されたデータ:")
    for mode in args.modes:
        data_dir = f"./data/{mode}_data"
        if os.path.exists(data_dir):
            file_count = len([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
            print(f"  - {data_dir}: {file_count} files")
        else:
            print(f"  - {data_dir}: 生成されませんでした")

def quick_generate(config_path: str, modes: list = None):
    """
    プログラムから呼び出し可能な簡単なデータ生成関数
    
    Args:
        config_path: 設定ファイルのパス
        modes: 生成するモード（デフォルト: ['train', 'val', 'test']）
    """
    if modes is None:
        modes = ['train', 'val', 'test']
    
    print(f"Generating data with config: {config_path}")
    
    # 設定の読み込み
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    # データ生成
    for mode in modes:
        print(f"Generating {mode} data...")
        create_data_files(config, data_mode=mode)
        print(f"✓ {mode} data generated")

if __name__ == "__main__":
    main()