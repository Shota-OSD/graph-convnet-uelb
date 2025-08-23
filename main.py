#!/usr/bin/env python3
"""
Graph Convolutional Network for UELB (Undirected Edge Load Balancing)
Refactored main script with modular architecture
"""

import sys
import argparse
from fastprogress import master_bar

from src.config.config_manager import ConfigManager
from src.data_management.dataset_manager import DatasetManager
from src.train.trainer import Trainer
from src.train.evaluator import Evaluator
from src.train.metrics import MetricsLogger, metrics_to_str

def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Graph Convolutional Network for UELB')
    parser.add_argument('--config', type=str, default='configs/default2.json',
                       help='設定ファイルのパス (default: configs/default2.json)')
    args = parser.parse_args()
    
    # 設定の初期化
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    dtypeFloat, dtypeLong = config_manager.get_dtypes()
    
    # データセット管理
    dataset_manager = DatasetManager(config)
    if dataset_manager.remake_dataset():
        dataset_manager.create_all_datasets()
    
    # トレーナーとエバリュエーターの初期化
    trainer = Trainer(config, dtypeFloat, dtypeLong)
    evaluator = Evaluator(config, dtypeFloat, dtypeLong)
    metrics_logger = MetricsLogger()
    
    # トレーニングパラメータ
    max_epochs = config.max_epochs
    val_every = config.val_every
    test_every = config.test_every
    learning_rate = config.learning_rate
    decay_rate = config.decay_rate
    
    # load_saved_modelがtrueの場合は評価のみ実行
    if config.get('load_saved_model', False):
        print("\n" + "="*60)
        print("MODEL EVALUATION MODE")
        print("="*60)
        print("Loaded saved model - Skipping training, running evaluation only\n")
        
        # 検証の実行
        print("Running validation...")
        val_time, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate = evaluator.evaluate(trainer.get_model(), None, mode='val')
        metrics_logger.log_val_metrics(val_approximation_rate, val_time)
        print('v: ' + metrics_to_str(0, val_time, learning_rate, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate))
        
        # テストの実行
        print("Running test...")
        test_time, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate = evaluator.evaluate(trainer.get_model(), None, mode='test')
        metrics_logger.log_test_metrics(test_approximation_rate, test_time)
        print('T: ' + metrics_to_str(0, test_time, learning_rate, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate))
        
        # 設定情報を辞書形式で準備（評価モード用）
        safe_config_info = {
            'config_file': args.config,
            'mode': 'evaluation_only',
            'model_hash': trainer._get_config_hash(),
            'num_val_data': config.get('num_val_data', 0),
            'num_test_data': config.get('num_test_data', 0),
        }
        
        # 評価結果のサマリーを表示
        metrics_logger.print_evaluation_summary(safe_config_info)
        
        # 評価結果をファイルに保存
        metrics_logger.save_results(safe_config_info)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        return
    
    # トレーニングループ
    epoch_bar = master_bar(range(max_epochs))
    for epoch in epoch_bar:
        # トレーニング
        train_time, train_loss, train_err_edges = trainer.train_one_epoch(epoch_bar)
        metrics_logger.log_train_metrics(train_loss, train_err_edges, train_time)
        epoch_bar.write(f"\nEpoch {epoch+1}/{max_epochs}")
        epoch_bar.write(f"Train - Loss: {train_loss:.4f}, Edge Error: {train_err_edges:.2f}%, Time: {train_time:.2f}s")
        
        # 検証
        if epoch % val_every == 0 or epoch == max_epochs - 1:
            val_time, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate = evaluator.evaluate(trainer.get_model(), epoch_bar, mode='val')
            metrics_logger.log_val_metrics(val_approximation_rate, val_time)
            epoch_bar.write('v: ' + metrics_to_str(epoch, val_time, learning_rate, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate))
        
        # 学習率の更新
        if epoch % val_every == 0 and epoch != 0:
            learning_rate /= decay_rate
            trainer.update_learning_rate(learning_rate)
            epoch_bar.write(f"Learning rate updated to: {learning_rate:.6f}")
        
        # テスト
        if epoch % test_every == 0 or epoch == max_epochs - 1:
            test_time, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate = evaluator.evaluate(trainer.get_model(), epoch_bar, mode='test')
            metrics_logger.log_test_metrics(test_approximation_rate, test_time)
            epoch_bar.write('T: ' + metrics_to_str(epoch, test_time, learning_rate, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate))
        
        # モデル保存
        if config.get('save_model', True):
            trainer.save_model(epoch, train_loss)
            
            # 古いモデルの削除
            if config.get('cleanup_old_models', True):
                trainer.cleanup_old_models()
    
    # 設定情報を辞書形式で準備（安全な形式のみ）
    safe_config_info = {
        'config_file': args.config,
        'mode': 'training',
        'model_hash': trainer._get_config_hash(),
        'max_epochs': max_epochs,
        'num_train_data': config.get('num_train_data', 0),
        'num_test_data': config.get('num_test_data', 0),
        'num_val_data': config.get('num_val_data', 0),
    }
    
    # 結果の出力
    metrics_logger.print_summary(safe_config_info)
    
    # 設定オブジェクトから安全に変換できる属性のみを追加
    try:
        config_dict = dict(config)
        # 基本的な設定値のみを抽出
        basic_config = {}
        for key, value in config_dict.items():
            if isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                basic_config[key] = value
        safe_config_info['basic_config'] = basic_config
    except Exception as e:
        print(f"Warning: Could not extract full config: {e}")
        safe_config_info['basic_config'] = {}
    
    # 結果をファイルに保存
    metrics_logger.save_results(safe_config_info)
    
    print("Training finished.")

if __name__ == "__main__":
    main() 