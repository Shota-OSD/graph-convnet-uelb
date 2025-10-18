import time
import torch
import torch.nn as nn
import numpy as np
import os
import hashlib
import json
from fastprogress import master_bar, progress_bar
from sklearn.utils.class_weight import compute_class_weight

from src.gcn.models.gcn_model import ResidualGatedGCNModel
from src.common.data_management.dataset_reader import DatasetReader
from src.gcn.models.model_utils import edge_error, update_learning_rate
from src.gcn.training.supervised_strategy import SupervisedLearningStrategy
from src.gcn.training.reinforcement_strategy import ReinforcementLearningStrategy

class Trainer:
    """トレーニングを担当するクラス"""
    
    def __init__(self, config, dtypeFloat, dtypeLong):
        self.config = config
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        self.models_dir = config.get('models_dir', './saved_models')
        self.net, self.optimizer = self._instantiate_model()

        # Initialize training strategy
        strategy_type = config.get('training_strategy', 'supervised')
        if strategy_type == 'supervised':
            self.strategy = SupervisedLearningStrategy(config)
            print(f"Using Supervised Learning Strategy")
        elif strategy_type == 'reinforcement':
            self.strategy = ReinforcementLearningStrategy(config)
            print(f"Using Reinforcement Learning Strategy (reward: {config.get('rl_reward_type', 'load_factor')})")
        else:
            raise ValueError(f"Unknown training strategy: {strategy_type}")
    
    def _instantiate_model(self):
        """モデルとオプティマイザーを初期化"""
        # 既存の保存済みモデルをチェック
        if self.config.get('load_saved_model', False):
            loaded_net, loaded_optimizer = self._try_load_saved_model()
            if loaded_net is not None:
                return loaded_net, loaded_optimizer
        
        net = nn.DataParallel(ResidualGatedGCNModel(self.config, self.dtypeFloat, self.dtypeLong))
        
        # GPU使用設定を確認
        use_gpu = self.config.get('use_gpu', True)
        if use_gpu and torch.cuda.is_available():
            net.cuda()
            print("Model moved to GPU")
        else:
            print("Model using CPU")
            
        nb_param = sum(np.prod(list(param.data.size())) for param in net.parameters())
        optimizer = torch.optim.Adam(net.parameters(), lr=self.config.learning_rate)
        
        # モデル情報の出力をコメントアウト
        # print('Number of parameters:', nb_param)
        # print(optimizer)
        torch.autograd.set_detect_anomaly(True)
        return net, optimizer
    
    def train_one_epoch(self, master_bar):
        """1エポックのトレーニングを実行"""
        self.net.train()
        mode = "train"
        num_data = self.config.get(f'num_{mode}_data')
        batch_size = self.config.batch_size
        accumulation_steps = self.config.accumulation_steps
        dataset = DatasetReader(num_data, batch_size, mode)

        batches_per_epoch = dataset.max_iter

        dataset = iter(dataset)
        running_loss = 0.0
        running_err_edges = 0.0
        running_nb_data = 0
        start_epoch = time.time()

        # Reset strategy metrics for new epoch
        self.strategy.reset_metrics()

        # Get device
        device = next(self.net.parameters()).device

        for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
            try:
                batch = next(dataset)
            except StopIteration:
                break

            # Prepare batch data using strategy's helper method
            batch_data = self.strategy.prepare_batch_data(batch, device)

            # Compute loss using selected strategy
            loss, metrics = self.strategy.compute_loss(self.net, batch_data, device)

            # Perform backward step using strategy
            self.strategy.backward_step(loss, self.optimizer, accumulation_steps, batch_num)

            # Update running metrics
            running_nb_data += batch_size
            running_loss += batch_size * loss.data.item()

            # Track edge error if available in metrics
            if 'edge_error' in metrics:
                running_err_edges += batch_size * metrics['edge_error']

        loss = running_loss / running_nb_data
        err_edges = running_err_edges / running_nb_data if running_nb_data > 0 else 0.0
        return time.time()-start_epoch, loss, err_edges
    
    def get_model(self):
        """モデルを取得"""
        return self.net
    
    def get_optimizer(self):
        """オプティマイザーを取得"""
        return self.optimizer
    
    def update_learning_rate(self, new_lr):
        """学習率を更新"""
        self.optimizer = update_learning_rate(self.optimizer, new_lr)
    
    def _get_config_hash(self):
        """設定のハッシュを生成してモデル識別に使用"""
        # すべての設定を含むハッシュ計算
        config_for_hash = {}
        
        # 設定オブジェクトから辞書に変換（Settingsはdictを継承）
        config_dict = dict(self.config)
        
        # ハッシュ計算に使用する設定を選択
        # モデル構造に影響する設定とデータ関連の設定を含める
        hash_keys = [
            # モデル構造関連
            'hidden_dim', 'num_layers', 'mlp_layers', 'node_dim',
            'voc_nodes_out', 'voc_edges_in', 'voc_edges_out',
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
    
    def _get_model_filename(self, epoch=None):
        """モデルファイル名を生成"""
        config_hash = self._get_config_hash()
        if epoch is not None:
            return f"model_{config_hash}_epoch_{epoch}.pt"
        else:
            return f"model_{config_hash}_latest.pt"
    
    def _try_load_saved_model(self):
        """保存済みモデルの読み込みを試行"""
        if not os.path.exists(self.models_dir):
            return None, None
        
        # 特定のエポックが指定されている場合はそれを優先
        load_epoch = self.config.get('load_model_epoch', None)
        if load_epoch is not None:
            model_filename = self._get_model_filename(epoch=load_epoch)
        else:
            model_filename = self._get_model_filename()
        
        model_path = os.path.join(self.models_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"No saved model found at {model_path}")
            # 利用可能なモデルを表示
            self._show_available_models()
            return None, None
        
        try:
            print(f"Loading saved model from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # モデル構造の検証
            expected_config = checkpoint.get('config_hash')
            current_config = self._get_config_hash()
            if expected_config != current_config:
                print(f"Config mismatch: expected {expected_config}, got {current_config}")
                return None, None
            
            # モデルとオプティマイザーの復元
            net = nn.DataParallel(ResidualGatedGCNModel(self.config, self.dtypeFloat, self.dtypeLong))
            net.load_state_dict(checkpoint['model_state_dict'])
            
            # GPU使用設定を確認
            use_gpu = self.config.get('use_gpu', True)
            if use_gpu and torch.cuda.is_available():
                net.cuda()
                print("Loaded model moved to GPU")
            else:
                print("Loaded model using CPU")
            
            optimizer = torch.optim.Adam(net.parameters(), lr=self.config.learning_rate)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Previous training loss: {checkpoint.get('loss', 'unknown')}")
            
            return net, optimizer
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None, None
    
    def save_model(self, epoch, loss, save_latest=True):
        """モデルを保存"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config_hash': self._get_config_hash(),
            'config': {
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'mlp_layers': self.config.mlp_layers,
                'node_dim': self.config.node_dim,
                'voc_nodes_in': self.config.num_commodities * 3,
                'voc_nodes_out': self.config.voc_nodes_out,
                'voc_edges_in': self.config.voc_edges_in,
                'voc_edges_out': self.config.voc_edges_out,
                'aggregation': self.config.aggregation,
                'dropout_rate': self.config.get('dropout_rate', 0.0),
            }
        }
        
        # エポック別保存
        if self.config.get('save_every_epoch', False):
            epoch_filename = self._get_model_filename(epoch)
            epoch_path = os.path.join(self.models_dir, epoch_filename)
            torch.save(checkpoint, epoch_path)
            print(f"Model saved to {epoch_path}")
        
        # 最新モデル保存
        if save_latest:
            latest_filename = self._get_model_filename()
            latest_path = os.path.join(self.models_dir, latest_filename)
            torch.save(checkpoint, latest_path)
            print(f"Latest model saved to {latest_path}")
    
    def cleanup_old_models(self, keep_last_n=5):
        """古いモデルファイルを削除"""
        if not os.path.exists(self.models_dir):
            return
        
        config_hash = self._get_config_hash()
        pattern = f"model_{config_hash}_epoch_"
        
        # エポック別ファイルを取得してソート
        epoch_files = []
        for filename in os.listdir(self.models_dir):
            if filename.startswith(pattern) and filename.endswith('.pt'):
                try:
                    epoch_str = filename[len(pattern):-3]  # .ptを除く
                    epoch_num = int(epoch_str)
                    epoch_files.append((epoch_num, filename))
                except ValueError:
                    continue
        
        # エポック番号でソート
        epoch_files.sort(key=lambda x: x[0])
        
        # 古いファイルを削除
        if len(epoch_files) > keep_last_n:
            for epoch_num, filename in epoch_files[:-keep_last_n]:
                filepath = os.path.join(self.models_dir, filename)
                os.remove(filepath)
                print(f"Removed old model: {filename}")
    
    def _show_available_models(self):
        """利用可能なモデルファイルを表示"""
        if not os.path.exists(self.models_dir):
            print("No saved models directory found.")
            return
        
        config_hash = self._get_config_hash()
        matching_models = []
        
        for filename in os.listdir(self.models_dir):
            if filename.startswith(f"model_{config_hash}_") and filename.endswith('.pt'):
                matching_models.append(filename)
        
        if matching_models:
            print("Available models for current configuration:")
            for model in sorted(matching_models):
                model_path = os.path.join(self.models_dir, model)
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    epoch = checkpoint.get('epoch', 'unknown')
                    loss = checkpoint.get('loss', 'unknown')
                    print(f"  - {model} (epoch: {epoch}, loss: {loss:.4f})" if isinstance(loss, (int, float)) else f"  - {model} (epoch: {epoch}, loss: {loss})")
                except:
                    print(f"  - {model} (info unavailable)")
        else:
            print(f"No models found for current configuration (hash: {config_hash})")
            # 他の設定のモデルがある場合は表示
            all_models = [f for f in os.listdir(self.models_dir) if f.startswith('model_') and f.endswith('.pt')]
            if all_models:
                print("Available models for other configurations:")
                for model in sorted(all_models)[:5]:  # 最大5個まで表示
                    print(f"  - {model}")
                if len(all_models) > 5:
                    print(f"  ... and {len(all_models) - 5} more")
    
    def list_available_models(self):
        """利用可能なモデルを一覧表示（外部から呼び出し可能）"""
        self._show_available_models()

    def train(self, evaluator, metrics_logger):
        """完全なトレーニングループを実行

        Args:
            evaluator: 評価器オブジェクト
            metrics_logger: メトリクス記録器

        Returns:
            net: 訓練済みモデル
        """
        from fastprogress import master_bar
        from ..train.metrics import metrics_to_str

        print("Starting training...")

        # トレーニングパラメータ
        max_epochs = self.config.max_epochs
        val_every = self.config.val_every
        test_every = self.config.test_every
        learning_rate = self.config.learning_rate
        decay_rate = self.config.decay_rate

        # マスターバーの初期化
        epoch_bar = master_bar(range(max_epochs))

        for epoch in epoch_bar:
            # トレーニング
            train_time, train_loss, train_err_edges = self.train_one_epoch(epoch_bar)
            metrics_logger.log_train_metrics(train_loss, train_err_edges, train_time)

            epoch_bar.write(f"\nEpoch {epoch+1}/{max_epochs}")
<<<<<<< Updated upstream
            epoch_bar.write(f"Train - Loss: {train_loss:.4f}, Edge Error: {train_err_edges:.2f}%, Time: {train_time:.2f}s")
=======

            # Display training metrics
            if rl_metrics:
                # RL training mode
                epoch_bar.write(f"Train - Loss: {train_loss:.4f}, Time: {train_time:.2f}s")
                if 'reward' in rl_metrics:
                    epoch_bar.write(f"  RL Metrics - Reward: {rl_metrics['reward']:.4f} (std: {rl_metrics.get('reward_std', 0):.4f}), "
                                  f"Advantage: {rl_metrics.get('advantage', 0):.4f} (std: {rl_metrics.get('advantage_std', 0):.4f}), "
                                  f"Entropy: {rl_metrics.get('entropy', 0):.4f}")
                if 'load_factor' in rl_metrics:
                    epoch_bar.write(f"  Load Factor: {rl_metrics['load_factor']:.4f}, Baseline: {rl_metrics.get('baseline', 0):.4f}")
                if 'policy_loss' in rl_metrics:
                    epoch_bar.write(f"  Policy Loss: {rl_metrics['policy_loss']:.4f}, Entropy Bonus: {rl_metrics.get('entropy_bonus', 0):.4f}")

                # Log RL metrics to file
                metrics_logger.log_rl_metrics(epoch, rl_metrics)
            else:
                # Supervised training mode
                epoch_bar.write(f"Train - Loss: {train_loss:.4f}, Edge Error: {train_err_edges:.2f}%, Time: {train_time:.2f}s")
>>>>>>> Stashed changes

            # 検証
            if epoch % val_every == 0 or epoch == max_epochs - 1:
                val_time, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate = evaluator.evaluate(
                    self.net, epoch_bar, mode='val'
                )
                metrics_logger.log_val_metrics(val_approximation_rate, val_time)

                epoch_bar.write('v: ' + metrics_to_str(epoch, val_time, learning_rate, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate))

            # 学習率の更新
            if epoch % val_every == 0 and epoch != 0:
                learning_rate /= decay_rate
                self.update_learning_rate(learning_rate)
                epoch_bar.write(f"Learning rate updated to: {learning_rate:.6f}")

            # テスト
            if epoch % test_every == 0 or epoch == max_epochs - 1:
                test_time, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate = evaluator.evaluate(
                    self.net, epoch_bar, mode='test'
                )
                metrics_logger.log_test_metrics(test_approximation_rate, test_time)

                epoch_bar.write('T: ' + metrics_to_str(epoch, test_time, learning_rate, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate))

            # モデル保存
            if self.config.get('save_model', True):
                self.save_model(epoch, train_loss)

                # 古いモデルの削除
                if self.config.get('cleanup_old_models', True):
                    self.cleanup_old_models()

        return self.net

 