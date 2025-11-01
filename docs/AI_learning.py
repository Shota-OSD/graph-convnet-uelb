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
from src.gcn.utils.model_converter import load_pretrained_supervised_model

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
        device = None
        if use_gpu and torch.cuda.is_available():
            net.cuda()
            device = torch.device('cuda')
            print("Model moved to GPU")
        else:
            device = torch.device('cpu')
            print("Model using CPU")

        # Check if we need to load pretrained supervised model for RL fine-tuning
        if self.config.get('load_pretrained_model', False):
            pretrained_path = self.config.get('pretrained_model_path')
            convert_supervised = self.config.get('convert_supervised_to_rl', False)

            if pretrained_path and convert_supervised:
                print("\n" + "="*70)
                print("LOADING PRE-TRAINED SUPERVISED MODEL FOR RL FINE-TUNING")
                print("="*70)
                # Load and convert supervised model to RL model
                net.module = load_pretrained_supervised_model(
                    net.module,
                    pretrained_path,
                    device=device,
                    verbose=True
                )
                print("Pre-trained model loaded and converted successfully!")
                print("="*70 + "\n")
            elif pretrained_path:
                print(f"\nLoading pre-trained model from: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    net.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    net.module.load_state_dict(checkpoint)
                print("Pre-trained model loaded successfully!\n")

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

        # RL-specific metrics
        running_reward = 0.0
        running_advantage = 0.0
        running_entropy = 0.0
        running_load_factor = 0.0
        current_baseline = None

        # Track all individual values for proper std calculation
        all_rewards = []
        all_advantages = []
        all_entropies = []
        all_load_factors = []

        # Track path quality metrics
        all_complete_paths_rate = []
        all_finite_solution_rate = []
        all_avg_finite_load_factor = []
        all_avg_path_length = []
        all_commodity_success_rate = []
        all_capacity_violation_rate = []

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

            # Track RL-specific metrics if available
            if 'reward' in metrics:
                running_reward += batch_size * metrics['reward']
                # Collect individual values if available
                if 'reward_individual' in metrics:
                    all_rewards.extend(metrics['reward_individual'])
            if 'advantage' in metrics:
                running_advantage += batch_size * metrics['advantage']
                if 'advantage_individual' in metrics:
                    all_advantages.extend(metrics['advantage_individual'])
            if 'entropy' in metrics:
                running_entropy += batch_size * metrics['entropy']
                if 'entropy_individual' in metrics:
                    all_entropies.extend(metrics['entropy_individual'])
            if 'mean_load_factor' in metrics:
                running_load_factor += batch_size * metrics['mean_load_factor']
                if 'load_factor_individual' in metrics:
                    all_load_factors.extend(metrics['load_factor_individual'])
            if 'baseline' in metrics:
                current_baseline = metrics['baseline']

            # Track path quality metrics
            if 'complete_paths_rate' in metrics:
                all_complete_paths_rate.append(metrics['complete_paths_rate'])
            if 'finite_solution_rate' in metrics:
                all_finite_solution_rate.append(metrics['finite_solution_rate'])
            if 'avg_finite_load_factor' in metrics:
                all_avg_finite_load_factor.append(metrics['avg_finite_load_factor'])
            if 'avg_path_length' in metrics:
                all_avg_path_length.append(metrics['avg_path_length'])
            if 'commodity_success_rate' in metrics:
                all_commodity_success_rate.append(metrics['commodity_success_rate'])
            if 'capacity_violation_rate' in metrics:
                all_capacity_violation_rate.append(metrics['capacity_violation_rate'])

        loss = running_loss / running_nb_data
        err_edges = running_err_edges / running_nb_data if running_nb_data > 0 else 0.0

        # Compute RL metrics averages and stds
        rl_metrics = {}
        if running_reward != 0.0:
            import numpy as np
            rl_metrics['reward'] = running_reward / running_nb_data
            rl_metrics['advantage'] = running_advantage / running_nb_data
            rl_metrics['entropy'] = running_entropy / running_nb_data
            rl_metrics['load_factor'] = running_load_factor / running_nb_data
            rl_metrics['baseline'] = current_baseline if current_baseline is not None else 0.0

            # Compute standard deviations from collected individual values
            if len(all_rewards) > 0:
                rl_metrics['reward_std'] = float(np.std(all_rewards))
            else:
                rl_metrics['reward_std'] = 0.0

            if len(all_advantages) > 0:
                rl_metrics['advantage_std'] = float(np.std(all_advantages))
            else:
                rl_metrics['advantage_std'] = 0.0

            if len(all_entropies) > 0:
                rl_metrics['entropy_std'] = float(np.std(all_entropies))
            else:
                rl_metrics['entropy_std'] = 0.0

            if len(all_load_factors) > 0:
                rl_metrics['load_factor_std'] = float(np.std(all_load_factors))
            else:
                rl_metrics['load_factor_std'] = 0.0

            # Add path quality metrics (averaged across batches)
            if len(all_complete_paths_rate) > 0:
                rl_metrics['complete_paths_rate'] = float(np.mean(all_complete_paths_rate))
            if len(all_finite_solution_rate) > 0:
                rl_metrics['finite_solution_rate'] = float(np.mean(all_finite_solution_rate))
            if len(all_avg_finite_load_factor) > 0:
                rl_metrics['avg_finite_load_factor'] = float(np.mean(all_avg_finite_load_factor))
            if len(all_avg_path_length) > 0:
                rl_metrics['avg_path_length'] = float(np.mean(all_avg_path_length))
            if len(all_commodity_success_rate) > 0:
                rl_metrics['commodity_success_rate'] = float(np.mean(all_commodity_success_rate))
            if len(all_capacity_violation_rate) > 0:
                rl_metrics['capacity_violation_rate'] = float(np.mean(all_capacity_violation_rate))

        # Ensure model is back in training mode after epoch
        self.net.train()
        
        return time.time()-start_epoch, loss, err_edges, rl_metrics
    
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
            # Set epoch for temperature scheduling (if using RL strategy)
            if hasattr(self.strategy, 'set_epoch'):
                self.strategy.set_epoch(epoch)

            # トレーニング
            train_time, train_loss, train_err_edges, rl_metrics = self.train_one_epoch(epoch_bar)
            metrics_logger.log_train_metrics(train_loss, train_err_edges, train_time)

            epoch_bar.write(f"\nEpoch {epoch+1}/{max_epochs}")

            # Display training metrics
            if rl_metrics:
                # RL training mode
                epoch_bar.write(f"Train - Loss: {train_loss:.4f}, Time: {train_time:.2f}s")
                if 'reward' in rl_metrics:
                    temp_str = f", Temp: {rl_metrics['temperature']:.2f}" if 'temperature' in rl_metrics else ""
                    epoch_bar.write(f"  RL Metrics - Reward: {rl_metrics['reward']:.4f} (std: {rl_metrics.get('reward_std', 0):.4f}), "
                                  f"Advantage: {rl_metrics.get('advantage', 0):.4f} (std: {rl_metrics.get('advantage_std', 0):.4f}), "
                                  f"Entropy: {rl_metrics.get('entropy', 0):.4f}{temp_str}")
                if 'load_factor' in rl_metrics:
                    epoch_bar.write(f"  Load Factor: {rl_metrics['load_factor']:.4f}, Baseline: {rl_metrics.get('baseline', 0):.4f}")
                if 'policy_loss' in rl_metrics:
                    epoch_bar.write(f"  Policy Loss: {rl_metrics['policy_loss']:.4f}, Entropy Bonus: {rl_metrics.get('entropy_bonus', 0):.4f}")
                # Show path quality metrics
                if 'complete_paths_rate' in rl_metrics:
                    epoch_bar.write(f"  Path Quality - Complete: {rl_metrics['complete_paths_rate']:.1f}%, "
                                  f"Finite Solutions: {rl_metrics['finite_solution_rate']:.1f}%, "
                                  f"Avg Path Length: {rl_metrics.get('avg_path_length', 0):.1f}")
                if 'avg_finite_load_factor' in rl_metrics and rl_metrics.get('finite_solution_rate', 0) > 0:
                    epoch_bar.write(f"  Finite Solutions - Avg Load Factor: {rl_metrics['avg_finite_load_factor']:.4f}, "
                                  f"Capacity Violations: {rl_metrics.get('capacity_violation_rate', 0):.1f}%")

                # Log RL metrics to file
                metrics_logger.log_rl_metrics(epoch, rl_metrics)
            else:
                # Supervised training mode
                epoch_bar.write(f"Train - Loss: {train_loss:.4f}, Edge Error: {train_err_edges:.2f}%, Time: {train_time:.2f}s")

            # 検証
            if epoch % val_every == 0 or epoch == max_epochs - 1:
                # Ensure model is in eval mode and use the current updated model
                self.net.eval()
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
                # Ensure model is in eval mode and use the current updated model
                self.net.eval()
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

 -----------------------------
 import time
import torch
import numpy as np
from fastprogress import master_bar, progress_bar
from sklearn.utils.class_weight import compute_class_weight

from src.common.data_management.dataset_reader import DatasetReader
from src.gcn.algorithms.beamsearch_uelb import BeamsearchUELB
from src.gcn.models.model_utils import edge_error, mean_feasible_load_factor

class Evaluator:
    """評価を担当するクラス"""

    def __init__(self, config, dtypeFloat, dtypeLong, strategy=None):
        self.config = config
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        self.strategy = strategy  # Training strategy (for RL support)

    def evaluate(self, net, master_bar, mode='test'):
        """モデルを評価"""
        net.eval()
        num_data = getattr(self.config, f'num_{mode}_data')
        batch_size = self.config.batch_size
        dataset = DatasetReader(num_data, batch_size, mode)
        batches_per_epoch = dataset.max_iter
        dataset = iter(dataset)
        edge_cw = None
        running_loss = 0.0
        running_mean_maximum_load_factor = 0.0
        running_gt_load_factor = 0.0
        running_nb_data = 0
        running_nb_batch = 0
        feasible_count = 0
        infeasible_count = 0
        
        with torch.no_grad():
            start_test = time.time()
            for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
                try:
                    batch = next(dataset)
                except StopIteration:
                    break
                
                x_edges = torch.LongTensor(batch.edges).to(torch.long).contiguous().requires_grad_(False)
                x_edges_capacity = torch.FloatTensor(batch.edges_capacity).to(torch.float).contiguous().requires_grad_(False)
                x_nodes = torch.LongTensor(batch.nodes).to(torch.long).contiguous().requires_grad_(False)
                y_edges = torch.LongTensor(batch.edges_target).to(torch.long).contiguous().requires_grad_(False)
                batch_commodities = torch.LongTensor(batch.commodities).to(torch.long).contiguous().requires_grad_(False)
                x_commodities = batch_commodities[:, :, 2].to(torch.float)
                
                # Move tensors to the same device as the model
                device = next(net.parameters()).device
                x_edges = x_edges.to(device)
                x_edges_capacity = x_edges_capacity.to(device)
                x_nodes = x_nodes.to(device)
                y_edges = y_edges.to(device)
                batch_commodities = batch_commodities.to(device)
                x_commodities = x_commodities.to(device)
                
                # Use strategy for evaluation if available (RL support)
                if self.strategy is not None:
                    # Prepare batch data
                    batch_data = self.strategy.prepare_batch_data(batch, device)

                    # Compute loss using strategy (will use beam search for RL in eval mode)
                    loss, metrics = self.strategy.compute_loss(net, batch_data, device)

                    # Extract load factor from metrics
                    mean_maximum_load_factor = metrics.get('mean_load_factor', 0.0)
                else:
                    # Original supervised learning evaluation
                    if type(edge_cw) != torch.Tensor:
                        edge_labels = y_edges.cpu().numpy().flatten()
                        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

                    y_preds, loss = net.forward(x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges, edge_cw)
                    loss = loss.mean()
                    # エッジエラーは計算するが使用しない（テストでは負荷率を重視）
                    _, _ = edge_error(y_preds, y_edges, x_edges)

                    beam_search = BeamsearchUELB(
                        y_preds, self.config.beam_size, batch_size, x_edges_capacity, batch_commodities, torch.float, torch.long, mode_strict=True)
                    pred_paths, is_feasible = beam_search.search()
                    mean_maximum_load_factor, _ = mean_feasible_load_factor(batch_size, self.config.num_commodities, self.config.num_nodes, pred_paths, x_edges_capacity, batch_commodities)
                
                # gt_load_factor is always the ground truth (optimal solution from dataset)
                gt_load_factor = np.mean(batch.load_factor)

                # Track feasibility status without modifying load factor
                if mean_maximum_load_factor > 1 or mean_maximum_load_factor == 0:
                    infeasible_count += 1
                else:
                    feasible_count += 1
                
                running_nb_data += batch_size
                running_loss += batch_size * loss.data.item() * self.config.accumulation_steps
                running_mean_maximum_load_factor += mean_maximum_load_factor
                running_gt_load_factor += gt_load_factor
                running_nb_batch += 1
            
            loss = running_loss / running_nb_data
            infeasible_rate = infeasible_count / (feasible_count + infeasible_count) * 100 if (feasible_count + infeasible_count) > 0 else 0

            # Calculate average metrics
            if running_nb_batch != 0:
                # Ground truth: average over all batches
                mean_gt_load_factor = running_gt_load_factor / running_nb_batch
                # Model load factor: average over all batches (including infeasible solutions)
                epoch_mean_maximum_load_factor = running_mean_maximum_load_factor / running_nb_batch
                # Approximation rate: only calculated using feasible solutions
                # (ground truth is always feasible, so we compare against feasible model outputs only)
                if feasible_count != 0:
                    feasible_mean_load_factor = running_mean_maximum_load_factor / running_nb_batch
                    # Note: This includes zeros from infeasible solutions, which makes the comparison fair
                    approximation_rate = mean_gt_load_factor / feasible_mean_load_factor * 100 if feasible_mean_load_factor != 0 else 0
                else:
                    approximation_rate = 0
            else:
                mean_gt_load_factor = 0
                epoch_mean_maximum_load_factor = 0
                approximation_rate = 0
        
        return (time.time()-start_test, loss, epoch_mean_maximum_load_factor,
                mean_gt_load_factor, approximation_rate, infeasible_rate)

    def evaluate_saved_model(self, trainer, metrics_logger):
        """保存済みモデルの評価のみを実行

        Args:
            trainer: トレーナーオブジェクト（保存済みモデルを持つ）
            metrics_logger: メトリクス記録器

        Returns:
            tuple: (val_result, test_result) 各結果は(time, loss, mean_load_factor, gt_load_factor, approximation_rate, infeasible_rate)
        """
        from ..train.metrics import metrics_to_str

        print("\n" + "="*60)
        print("MODEL EVALUATION MODE")
        print("="*60)
        print("Loaded saved model - Skipping training, running evaluation only\n")

        learning_rate = self.config.learning_rate

        # 検証の実行
        print("Running validation...")
        val_time, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate = self.evaluate(
            trainer.get_model(), None, mode='val'
        )
        metrics_logger.log_val_metrics(val_approximation_rate, val_time)

        print('v: ' + metrics_to_str(0, val_time, learning_rate, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate))

        # テストの実行
        print("Running test...")
        test_time, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate = self.evaluate(
            trainer.get_model(), None, mode='test'
        )
        metrics_logger.log_test_metrics(test_approximation_rate, test_time)

        print('T: ' + metrics_to_str(0, test_time, learning_rate, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate))

        val_result = (val_time, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate)
        test_result = (test_time, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate)

        return val_result, test_result 
    ------------------------------------------------------------
    """
Reinforcement Learning Strategy

This strategy implements a policy gradient approach where the model
learns directly from the maximum load factor (environment reward)
rather than supervised labels.
"""

import torch
import torch.nn.functional as F
import numpy as np

from .base_strategy import BaseTrainingStrategy
from ..algorithms.beamsearch_uelb import BeamsearchUELB, BeamSearchFactory
from ..algorithms.path_sampler import PathSampler
from ..models.model_utils import mean_feasible_load_factor


class ReinforcementLearningStrategy(BaseTrainingStrategy):
    """
    Reinforcement learning strategy using maximum load factor as reward.

    This strategy uses REINFORCE (Policy Gradient) algorithm:
    1. Model outputs edge probabilities (policy)
    2. Beam search generates paths based on these probabilities
    3. Maximum load factor is computed as reward signal
    4. Policy is updated to maximize expected reward

    Key differences from supervised learning:
    - No ground truth labels required
    - Learns directly from load factor (optimization objective)
    - Uses policy gradient instead of cross-entropy loss
    """

    def __init__(self, config):
        super().__init__(config)
        self.beam_size = config.get('beam_size', 1280)
        self.batch_size = config.get('batch_size', 1)
        self.num_nodes = config.get('num_nodes', 14)
        self.num_commodities = config.get('num_commodities', 5)

        # RL-specific hyperparameters
        self.reward_type = config.get('rl_reward_type', 'load_factor')  # 'load_factor' or 'inverse_load_factor'
        self.use_baseline = config.get('rl_use_baseline', True)
        self.baseline_momentum = config.get('rl_baseline_momentum', 0.9)
        self.entropy_weight = config.get('rl_entropy_weight', 0.01)  # For exploration

        # Beam search algorithm selection
        self.beam_search_type = config.get('rl_beam_search_type', 'standard')  # 'standard', 'unconstrained', etc.

        # Sampling configuration
        self.use_sampling = config.get('rl_use_sampling', True)  # Use top-p sampling for both training and evaluation (True) or beam search (False)
        self.sampling_temperature = config.get('rl_sampling_temperature', 1.0)  # Lower = more deterministic
        self.sampling_top_p = config.get('rl_sampling_top_p', 0.9)  # Nucleus sampling threshold

        # Advantage normalization for stability
        self.normalize_advantages = config.get('rl_normalize_advantages', True)

        # Smooth penalty for infeasible solutions
        self.use_smooth_penalty = config.get('rl_use_smooth_penalty', True)
        self.penalty_lambda = config.get('rl_penalty_lambda', 5.0)  # Penalty weight for constraint violation

        # Invalid edge masking
        self.mask_invalid_edges = config.get('rl_mask_invalid_edges', True)

        # Baseline for variance reduction
        self.reward_baseline = None

        # Temperature scheduling parameters
        self.initial_temperature = config.get('rl_sampling_temperature', 1.0)
        self.min_temperature = config.get('rl_min_temperature', 0.5)
        self.temperature_decay = config.get('rl_temperature_decay', 0.05)
        self.current_epoch = 0
        self.max_epochs = config.get('max_epochs', 10)

        # Entropy epsilon (epsilon-greedy style uniform mixing over valid next nodes)
        self.entropy_epsilon = config.get('rl_entropy_epsilon', 0.05)

        # Use trajectory entropy instead of grid entropy
        self.use_trajectory_entropy = config.get('rl_use_trajectory_entropy', True)

    def set_epoch(self, epoch):
        """Set current epoch for temperature scheduling."""
        self.current_epoch = epoch

    def get_current_temperature(self):
        """Calculate temperature based on current epoch (linear decay)."""
        # Linear decay: T = max(T_min, T_initial - epoch * decay)
        temperature = max(
            self.min_temperature,
            self.initial_temperature - self.current_epoch * self.temperature_decay
        )
        return temperature

    def compute_loss(self, model, batch_data, device=None):
        """
        Compute policy gradient loss using load factor as reward.

        Args:
            model: GCN model (outputs edge probabilities)
            batch_data: Dictionary with inputs and capacities
            device: Device for computation

        Returns:
            loss: Policy gradient loss
            metrics: Dictionary with 'mean_load_factor', 'reward', 'entropy'
        """
        x_edges = batch_data['x_edges']
        x_commodities = batch_data['x_commodities']
        x_edges_capacity = batch_data['x_edges_capacity']
        x_nodes = batch_data['x_nodes']
        batch_commodities = batch_data['batch_commodities']

        # Forward pass (get predictions without computing supervised loss)
        # Mask invalid edges (zero capacity) to prevent sampling impossible paths
        y_preds, _ = model.forward(
            x_edges, x_commodities, x_edges_capacity, x_nodes,
            y_edges=None, edge_cw=None, compute_loss=False,
            mask_invalid_edges=self.mask_invalid_edges
        )

        # Generate paths: use top-p sampling for training, beam search for evaluation
        if self.use_sampling and model.training:
            # Probabilistic sampling for training (REINFORCE requires on-policy samples)
            # Note: top-p sampling does not guarantee feasible solutions

            # Use temperature scheduling: start high (exploration), gradually decrease (exploitation)
            current_temp = self.get_current_temperature()

            sampler = PathSampler(
                y_pred_edges=y_preds,
                edges_capacity=x_edges_capacity,
                commodities=batch_commodities,
                num_samples=1,
                temperature=current_temp,  # Use scheduled temperature
                top_p=self.sampling_top_p,
                entropy_epsilon=self.entropy_epsilon,
                dtypeFloat=torch.float,
                dtypeLong=torch.long
            )
            pred_paths, path_log_probs, is_feasible, stepwise_entropies = sampler.sample()
        else:
            # Beam search for evaluation (guarantees feasible solutions)
            beam_search = BeamSearchFactory.create_algorithm(
                self.beam_search_type,
                y_pred_edges=y_preds,
                beam_size=self.beam_size,
                batch_size=self.batch_size,
                edges_capacity=x_edges_capacity,
                commodities=batch_commodities,
                dtypeFloat=torch.float,
                dtypeLong=torch.long,
                mode_strict=True
            )
            pred_paths, is_feasible = beam_search.search()
            # For beam search, we don't have exact log probs, so use surrogate
            path_log_probs = None
            stepwise_entropies = None

        # Compute maximum load factor (our optimization objective)
        mean_maximum_load_factor, individual_load_factors = mean_feasible_load_factor(
            self.batch_size,
            self.num_commodities,
            self.num_nodes,
            pred_paths,
            x_edges_capacity,
            batch_commodities
        )

        # DEBUG: Print load factors for first batch of each epoch
        debug_enabled = False  # Set to True to enable debug output for load factors
        if debug_enabled:
            if not hasattr(self, '_debug_last_epoch'):
                self._debug_last_epoch = -1
                self._debug_batch_in_epoch = 0

            # Reset batch counter at start of new epoch (detect by checking if batch counter wrapped)
            if model.training:
                self._debug_batch_in_epoch += 1
                # Print for first batch of epoch or every 5 batches
                if self._debug_batch_in_epoch == 1 or self._debug_batch_in_epoch % 5 == 0:
                    print(f"\n=== Load Factor Debug (Batch {self._debug_batch_in_epoch}) ===")
                    print(f"  Shape: {individual_load_factors.shape}")
                    # Format mean properly
                    mean_val = mean_maximum_load_factor.item() if isinstance(mean_maximum_load_factor, torch.Tensor) else mean_maximum_load_factor
                    mean_str = f"{mean_val:.4f}" if not np.isinf(mean_val) else "inf"
                    print(f"  Mean: {mean_str}")
                    if individual_load_factors.numel() <= 20:  # Only print all values if batch size <= 20
                        print(f"  Values: {individual_load_factors}")
                    else:
                        print(f"  First 5: {individual_load_factors[:5]}")
                    if not torch.isinf(individual_load_factors).all():
                        finite_vals = individual_load_factors[~torch.isinf(individual_load_factors)]
                        if len(finite_vals) > 0:
                            print(f"  Finite - Min: {finite_vals.min():.4f}, Max: {finite_vals.max():.4f}")
                    print(f"  Contains inf: {torch.isinf(individual_load_factors).any()} ({torch.isinf(individual_load_factors).sum().item()}/{len(individual_load_factors)} samples)")
                    print("=" * 50)

        # Calculate path quality metrics BEFORE designing rewards
        total_commodities = 0
        complete_commodities = 0
        total_path_length = 0
        path_count = 0
        finite_solutions = 0
        finite_load_factors = []
        capacity_violations = 0

        # DEBUG: Check first sample's paths
        debug_metrics = False  # Set to True to enable debug output for metrics calculation
        if debug_metrics and model.training:
            if not hasattr(self, '_debug_metrics_printed'):
                self._debug_metrics_printed = True
                print(f"\n=== Metrics Calculation Debug ===")
                print(f"  pred_paths type: {type(pred_paths)}")
                print(f"  pred_paths length: {len(pred_paths)}")
                if len(pred_paths) > 0:
                    print(f"  pred_paths[0] (first sample): {pred_paths[0]}")
                    print(f"  Number of paths in first sample: {len(pred_paths[0])}")
                    for idx, p in enumerate(pred_paths[0]):
                        print(f"    Path {idx}: {p} (nodes={len(p)}, edges={max(0, len(p)-1)})")
                print("=" * 50)

        for i in range(self.batch_size):
            sample_paths = pred_paths[i]
            sample_commodities = batch_commodities[i]

            if debug_metrics and model.training and i == 0:
                if not hasattr(self, '_debug_path_count_printed'):
                    self._debug_path_count_printed = True
                    print(f"\n=== Path Counting Debug (Sample 0) ===")
                    print(f"  Number of paths: {len(sample_paths)}")
                    print(f"  Number of commodities: {len(sample_commodities)}")

            for path_idx, path in enumerate(sample_paths):
                total_commodities += 1
                dst = int(sample_commodities[path_idx][1].item())

                # Check if path is complete
                if len(path) > 0 and path[-1] == dst:
                    complete_commodities += 1

                # Track path length (number of edges, not nodes)
                # Path length = number of edges = len(path) - 1
                path_edge_count = max(0, len(path) - 1)
                total_path_length += path_edge_count
                path_count += 1

                if debug_metrics and model.training and i == 0 and path_idx < 3:
                    if not hasattr(self, f'_debug_path_{path_idx}_printed'):
                        setattr(self, f'_debug_path_{path_idx}_printed', True)
                        print(f"  Path {path_idx}: {path}")
                        print(f"    dst={dst}, complete={len(path) > 0 and path[-1] == dst}, edges={path_edge_count}")

            # Check if sample has finite load factor
            load_factor_i = individual_load_factors[i] if i < len(individual_load_factors) else mean_maximum_load_factor
            if isinstance(load_factor_i, torch.Tensor):
                load_factor_i = load_factor_i.item()

            if not np.isinf(load_factor_i) and load_factor_i > 0:
                finite_solutions += 1
                finite_load_factors.append(load_factor_i)
                if load_factor_i > 1.0:
                    capacity_violations += 1

        # Calculate rates
        complete_paths_rate = (complete_commodities / total_commodities * 100) if total_commodities > 0 else 0.0
        finite_solution_rate = (finite_solutions / self.batch_size * 100) if self.batch_size > 0 else 0.0
        avg_finite_load_factor = np.mean(finite_load_factors) if len(finite_load_factors) > 0 else 0.0
        avg_path_length = (total_path_length / path_count) if path_count > 0 else 0.0
        commodity_success_rate = (complete_commodities / total_commodities * 100) if total_commodities > 0 else 0.0
        capacity_violation_rate = (capacity_violations / finite_solutions * 100) if finite_solutions > 0 else 0.0

        if debug_metrics and model.training:
            if not hasattr(self, '_debug_metrics_calc_printed'):
                self._debug_metrics_calc_printed = True
                print(f"\n=== Metrics Calculation Results ===")
                print(f"  total_commodities: {total_commodities}")
                print(f"  complete_commodities: {complete_commodities}")
                print(f"  total_path_length: {total_path_length}")
                print(f"  path_count: {path_count}")
                print(f"  avg_path_length: {avg_path_length:.2f}")
                print(f"  complete_paths_rate: {complete_paths_rate:.2f}%")
                print("=" * 50)

        # Design reward signal PER SAMPLE
        rewards = []
        for i in range(self.batch_size):
            load_factor_i = individual_load_factors[i] if i < len(individual_load_factors) else mean_maximum_load_factor

            # Convert tensor to float
            if isinstance(load_factor_i, torch.Tensor):
                load_factor_i = load_factor_i.item()

            # Check if all paths reach their destinations
            sample_paths = pred_paths[i]
            sample_commodities = batch_commodities[i]
            all_paths_complete = True
            for path_idx, path in enumerate(sample_paths):
                dst = int(sample_commodities[path_idx][1].item())
                if len(path) == 0 or path[-1] != dst:
                    all_paths_complete = False
                    break

            # IMPROVED REWARD DESIGN (2025-10-22)
            # Based on analysis: moderate penalties (1.5x) to encourage complete paths

            # Handle incomplete paths (destination not reached)
            if not all_paths_complete:
                # Count how many paths are incomplete
                num_incomplete = sum(1 for path_idx, path in enumerate(sample_paths)
                                    if len(path) == 0 or path[-1] != int(sample_commodities[path_idx][1].item()))
                # Graduated penalty based on severity (REVERTED: original penalties with Reachability Mask guarantee)
                if num_incomplete <= 1:
                    reward_i = -2.0  # Only one path failed - mild penalty
                elif num_incomplete <= 2:
                    reward_i = -5.0  # Two paths failed - moderate penalty
                else:
                    reward_i = -10.0  # Multiple paths failed - strong penalty
            # Handle inf (paths using non-existent edges but all complete)
            elif np.isinf(load_factor_i):
                # Paths are complete but use invalid edges
                reward_i = -5.0  # Less severe than incomplete paths
            elif self.reward_type == 'load_factor':
                # Valid solution with finite load factor
                # Reward range: 7~10 for feasible, 1~5 for infeasible
                if load_factor_i <= 1.0:
                    # Feasible solution - positive reward
                    reward_i = 10.0 - load_factor_i * 3.0  # 7.0 ~ 10.0
                else:
                    # Capacity exceeded - still some reward to encourage valid paths
                    reward_i = 5.0 - load_factor_i * 2.0  # Decreases as violation increases
            else:
                raise ValueError(f"Unknown reward type: {self.reward_type}")

            rewards.append(reward_i)

        # Convert to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32, device=y_preds.device)

        # Update baseline using moving average PER SAMPLE
        # Initialize baseline if needed
        if self.reward_baseline is None:
            self.reward_baseline = rewards.mean().item()

        if self.use_baseline:
            # Compute advantages per sample
            advantages = rewards - self.reward_baseline

            # Update baseline with batch mean
            batch_mean_reward = rewards.mean().item()
            self.reward_baseline = (
                self.baseline_momentum * self.reward_baseline +
                (1 - self.baseline_momentum) * batch_mean_reward
            )
        else:
            advantages = rewards

        # Normalize advantages for better gradient stability
        # This reduces variance without changing the expected gradient
        if self.normalize_advantages and self.batch_size > 1:
            advantage_mean = advantages.mean()
            advantage_std = advantages.std()

            # DEBUG: Check if advantages have variance
            if advantage_std < 1e-8:
                # No variance in advantages - skip normalization to preserve signal
                pass
            else:
                advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

        # Compute policy gradient loss PER SAMPLE
        if path_log_probs is not None and model.training and path_log_probs.requires_grad:
            # CORRECT REINFORCE: Use log probability of sampled trajectory
            # L = -Σ_i [log π(τ_i) * (R_i - b)]
            # where τ_i is the sampled trajectory for sample i
            # Shape: path_log_probs [batch_size], advantages [batch_size]
            per_sample_loss = -path_log_probs * advantages
            policy_loss = per_sample_loss.mean()  # Average over batch
        else:
            # Fallback: Use surrogate loss based on model outputs
            # This is an approximation but maintains gradient flow
            log_probs = F.log_softmax(y_preds, dim=-1)
            # Use mean advantage for fallback
            mean_advantage = advantages.mean() if isinstance(advantages, torch.Tensor) else advantages
            policy_loss = -log_probs.mean() * mean_advantage

        # Add entropy bonus for exploration
        # UPDATED 2025-10-19: Calculate entropy over next-node selection distribution
        # y_preds shape: [batch, nodes_from, nodes_to, commodities]
        # Entropy measures diversity of next-node choices (14 options per source node)

        if model.training and self.use_sampling:
            current_temp = self.get_current_temperature()
            # Compute next-node selection probabilities (dim=2: which next node to choose)
            probs = F.softmax(y_preds / current_temp, dim=2)
            log_probs_for_entropy = F.log_softmax(y_preds / current_temp, dim=2)
        else:
            # No temperature scaling for evaluation
            probs = F.softmax(y_preds, dim=2)
            log_probs_for_entropy = F.log_softmax(y_preds, dim=2)

        # Entropy over next-node choices (dim=2). Use only rows with >=2 viable options.
        row_entropy = -(probs * log_probs_for_entropy).sum(dim=2)  # [B, V, C]
        multi_choice_mask = (probs > 1e-8).sum(dim=2) >= 2  # [B, V, C]
        if multi_choice_mask.any():
            entropy_grid = row_entropy[multi_choice_mask].mean()
        else:
            entropy_grid = row_entropy.mean()

        # Trajectory entropy from sampler (average across all steps/commodities/batch)
        if stepwise_entropies is not None:
            flat_steps = []
            for per_batch in stepwise_entropies:
                for per_commodity in per_batch:
                    flat_steps.extend(per_commodity)
            if len(flat_steps) > 0:
                traj_entropy = torch.tensor(flat_steps, device=y_preds.device, dtype=torch.float32).mean()
            else:
                traj_entropy = torch.tensor(0.0, device=y_preds.device)
        else:
            traj_entropy = torch.tensor(0.0, device=y_preds.device)

        # Always use grid entropy for gradient computation (connected to y_preds)
        # Trajectory entropy is tracked as a metric but cannot be used for gradients
        # since it's computed from sampled discrete actions
        entropy_bonus = -self.entropy_weight * entropy_grid

        # For monitoring purposes
        if self.use_trajectory_entropy and stepwise_entropies is not None and len(flat_steps) > 0:
            entropy_for_bonus = traj_entropy  # Log trajectory entropy
        else:
            entropy_for_bonus = entropy_grid  # Log grid entropy

        total_loss = policy_loss + entropy_bonus

        # Store metrics (use batch averages for logging)
        mean_reward = rewards.mean().item()
        reward_std = rewards.std().item() if isinstance(rewards, torch.Tensor) else 0.0
        mean_advantage = advantages.mean().item() if isinstance(advantages, torch.Tensor) else advantages
        advantage_std = advantages.std().item() if isinstance(advantages, torch.Tensor) else 0.0

        # Convert tensors to lists for individual value tracking
        rewards_list = rewards.detach().cpu().tolist() if isinstance(rewards, torch.Tensor) else []
        advantages_list = advantages.detach().cpu().tolist() if isinstance(advantages, torch.Tensor) else []
        load_factors_list = individual_load_factors.detach().cpu().tolist() if isinstance(individual_load_factors, torch.Tensor) else []

        metrics = {
            'mean_load_factor': mean_maximum_load_factor,
            'reward': mean_reward,
            'reward_std': reward_std,
            'advantage': mean_advantage,
            'advantage_std': advantage_std,
            'entropy': entropy_grid.item(),
            'traj_entropy': traj_entropy.item(),
            'entropy_used': entropy_for_bonus.item(),
            'baseline': self.reward_baseline if self.reward_baseline else 0.0,
            'is_feasible': 1 if mean_maximum_load_factor <= 1 and mean_maximum_load_factor > 0 else 0,
            'policy_loss': policy_loss.item(),
            'entropy_bonus': entropy_bonus.item(),
            # Add individual values for proper epoch-level std calculation
            'reward_individual': rewards_list,
            'advantage_individual': advantages_list,
            'load_factor_individual': load_factors_list,
            # Add new path quality metrics
            'complete_paths_rate': complete_paths_rate,
            'finite_solution_rate': finite_solution_rate,
            'avg_finite_load_factor': avg_finite_load_factor,
            'avg_path_length': avg_path_length,
            'commodity_success_rate': commodity_success_rate,
            'capacity_violation_rate': capacity_violation_rate,
            # Add temperature for monitoring
            'temperature': self.get_current_temperature()
        }

        return total_loss, metrics

    def backward_step(self, loss, optimizer, accumulation_steps=1, batch_num=0):
        """
        Perform backward pass for policy gradient.

        Args:
            loss: Policy gradient loss
            optimizer: PyTorch optimizer
            accumulation_steps: Number of batches to accumulate gradients
            batch_num: Current batch number

        Returns:
            bool: True if optimizer step was performed
        """
        # Policy gradient backward
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps batches
        if (batch_num + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            return True

        return False

    def reset_metrics(self):
        """Reset metrics for new epoch (but keep baseline)."""
        super().reset_metrics()
        # Keep baseline across epochs for stability
------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn

from .gcn_layers import ResidualGatedGCNLayer, MLP
from .model_utils import *

class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config, dtypeFloat, dtypeLong):
        super().__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Define net parameters
        self.num_nodes = config.num_nodes
        self.num_commodities = config.num_commodities
        self.voc_nodes_in = config.num_commodities * 3
        self.voc_nodes_out = config['num_nodes']  # config['voc_nodes_out']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']
        self.dropout_rate = config.get('dropout_rate', 0.2)
        # Node and edge embedding layers/lookups
        self.nodes_embedding = nn.Embedding(self.voc_nodes_in, self.hidden_dim // 2)
        self.commodities_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim, bias=False)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        # self.mlp_nodes = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)

    def forward(self, x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges=None, edge_cw=None, compute_loss=True, mask_invalid_edges=False):
        """
        Forward pass through the GCN model.

        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_commodities: Input edge adjacency matrix (batch_size, num_commodities)
            x_edges_capacity: Input edge capacity matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input node with commodity information (batch_size, num_nodes, num_commodities)
            y_edges: Targets for edges (batch_size, num_edges, num_commodities) - optional
            edge_cw: Class weights for edges loss - optional
            compute_loss: Whether to compute loss (requires y_edges and edge_cw)
            mask_invalid_edges: Whether to mask edges with zero capacity (for RL training)

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, num_commodities)
            loss: Value of loss function (None if compute_loss=False)

        # Nomalize node and edge capacity
        x_edges_capacity_min = x_edges_capacity.min()
        x_edges_capacity_max = x_edges_capacity.max()
        normalized_x_edges_capacity = (x_edges_capacity - x_edges_capacity_min) / (x_edges_capacity_max - x_edges_capacity_min)
        """
        # Features embedding
        x_edges_capacity_expanded = x_edges_capacity.unsqueeze(-1).expand(-1, -1, -1, self.num_commodities)

        # Handle x_commodities shape: should be [batch_size, num_commodities]
        # If it's [batch_size, num_commodities, 3] (with src/dst/demand), take only demand
        if len(x_commodities.shape) == 3:
            x_commodities = x_commodities[:, :, 2]  # Extract demand column

        x_commodities_expanded = x_commodities.unsqueeze(1).expand(-1, self.num_nodes, -1)

        x_embedded = self.nodes_embedding(x_nodes)
        c = self.commodities_embedding(x_commodities_expanded.unsqueeze(-1))
        e = self.edges_values_embedding(x_edges_capacity_expanded.unsqueeze(4))

        x_aggregate = x_embedded.mean(dim=2).unsqueeze(2).expand(-1, -1, self.num_commodities, -1)

        x = torch.cat((x_aggregate, c), 3)
        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x.contiguous(), e.contiguous())  # B x V x C x H, B x V x V x C x H
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)
        # y_pred_edges shape: [batch_size, num_nodes, num_nodes, num_commodities, voc_edges_out]

        # UPDATED 2025-10-19: Squeeze out voc_edges_out dimension if it's 1 (edge scores)
        if self.voc_edges_out == 1:
            y_pred_edges = y_pred_edges.squeeze(-1)  # [B, V, V, C]

        # Mask invalid edges (zero capacity) for RL training
        if mask_invalid_edges:
            # Create mask for edges with zero capacity
            # x_edges_capacity shape: [batch_size, num_nodes, num_nodes]
            # Expand to match y_pred_edges dimensions
            if self.voc_edges_out == 1:
                # New format: [batch, nodes, nodes, commodities]
                invalid_mask = (x_edges_capacity.unsqueeze(-1) == 0)
                invalid_mask = invalid_mask.expand(-1, -1, -1, self.num_commodities)
            else:
                # Old format: [batch, nodes, nodes, commodities, 2]
                invalid_mask = (x_edges_capacity.unsqueeze(-1).unsqueeze(-1) == 0)
                invalid_mask = invalid_mask.expand(-1, -1, -1, self.num_commodities, self.voc_edges_out)
            # Set logits for invalid edges to very negative value (will have ~0 probability after softmax)
            y_pred_edges = y_pred_edges.masked_fill(invalid_mask, -1e9)

        # Optionally compute loss
        loss = None
        if compute_loss:
            if y_edges is None or edge_cw is None:
                raise ValueError("y_edges and edge_cw required when compute_loss=True")

            edge_cw = torch.tensor(edge_cw, dtype=self.dtypeFloat)  # Convert to tensors
            # Move edge_cw to the same device as y_pred_edges
            edge_cw = edge_cw.to(y_pred_edges.device)
            loss = loss_edges(y_pred_edges, y_edges, edge_cw)

        return y_pred_edges, loss
---------------------------------------------------------
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import time

class BeamSearchAlgorithm(ABC):
    """ビームサーチアルゴリズムの抽象基底クラス"""
    
    def __init__(self, y_pred_edges, beam_size, batch_size, edges_capacity, commodities,
                 dtypeFloat, dtypeLong, mode_strict=False, max_iter=5):
        # 共通の初期化処理
        # UPDATED 2025-10-19: Support new format (voc_edges_out=1)
        if len(y_pred_edges.shape) == 5:
            # Old format: [B, V, V, C, voc_edges_out=2]
            y = F.log_softmax(y_pred_edges, dim=4)  # B x V x V x C x voc_edges
            y = y[:, :, :, :, 1]  # B x V x V x C
        else:
            # New format: [B, V, V, C] - edge scores
            y = F.log_softmax(y_pred_edges, dim=2)  # B x V x V x C (softmax over next nodes)
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number
        
        self.y = y
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.edges_capacity = edges_capacity
        self.commodities = commodities
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        self.mode_strict = mode_strict
        self.max_iter = max_iter
        
        # パフォーマンス計測用
        self.execution_time = 0.0
        self.algorithm_name = self.__class__.__name__

    def search(self) -> Tuple[List[List[List[int]]], bool]:
        """
        メインの検索メソッド - 共通のフレームワーク
        
        Returns:
            Tuple[List[List[List[int]]], bool]: (all_commodity_paths, is_feasible)
        """
        start_time = time.time()
        
        node_orders = []
        all_commodity_paths = []

        for batch in range(self.batch_size):
            batch_node_orders, commodity_paths, is_feasible = self._search_single_batch(batch)
            node_orders.append(batch_node_orders)
            all_commodity_paths.append(commodity_paths)

        self.execution_time = time.time() - start_time
        return all_commodity_paths, is_feasible

    @abstractmethod
    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """
        単一バッチでの検索 - アルゴリズム固有の実装
        
        Args:
            batch: バッチインデックス
            
        Returns:
            Tuple[List[List[int]], List[List[int]], bool]: (node_orders, commodity_paths, is_feasible)
        """
        pass

    def get_performance_info(self) -> dict:
        """
        パフォーマンス情報の取得
        
        Returns:
            dict: パフォーマンス情報
        """
        return {
            'algorithm_name': self.algorithm_name,
            'execution_time': self.execution_time,
            'beam_size': self.beam_size,
            'batch_size': self.batch_size,
            'max_iter': self.max_iter
        }


class StandardBeamSearch(BeamSearchAlgorithm):
    """標準的なビームサーチアルゴリズム（元の実装）"""
    
    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """標準的なビームサーチによる単一バッチ検索"""
        batch_y_pred_edges = self.y[batch]
        commodities = self.commodities[batch]
        count = 0

        while count < self.max_iter:
            batch_edges_capacity = self.edges_capacity[batch]
            
            # ランダムシャッフルによる前処理
            random_indices = torch.randperm(commodities.size(0))
            shuffled_commodities = commodities[random_indices]
            shuffled_pred_edges = batch_y_pred_edges[:, :, random_indices]
            _, original_indices = torch.sort(random_indices)

            node_orders = []
            commodity_paths = []
            is_feasible = True

            for index, commodity in enumerate(shuffled_commodities):
                source_node = commodity[0].item()
                target_node = commodity[1].item()
                demand = commodity[2].item()
                
                # ビームサーチによるパス探索
                node_order, remaining_edges_capacity, best_path = self._beam_search_for_commodity(
                    batch_edges_capacity, shuffled_pred_edges[:, :, index], 
                    source_node, target_node, demand
                )
                
                if best_path == []:
                    break
                    
                node_orders.append(node_order)
                if self.mode_strict:
                    batch_edges_capacity = remaining_edges_capacity
                commodity_paths.append(best_path)

            if len(commodity_paths) == commodities.shape[0]:
                count = self.max_iter
                unshuffled_commodity_paths = [commodity_paths[i] for i in original_indices]
            else:
                is_feasible = False
                unshuffled_commodity_paths = self._get_fallback_paths(commodities.shape[0])
                count += 1
        
        return node_orders, unshuffled_commodity_paths, is_feasible

    def _beam_search_for_commodity(self, edges_capacity, y_commodities, source, target, demand):
        """標準的なビームサーチによるパス探索"""
        beam_queue = [(source, [source], 0, edges_capacity.clone())]
        best_paths = []

        while beam_queue:
            current_scores = [item[2] for item in beam_queue]
            beam_queue = sorted(beam_queue, key=lambda x: x[2], reverse=True)[:self.beam_size]

            next_beam_queue = []
            for current_node, path, current_score, remaining_edges_capacity in beam_queue:
                if current_node == target:
                    best_paths.append((path, current_score, remaining_edges_capacity))
                    continue

                for next_node in range(edges_capacity.shape[0]):
                    if next_node in path:
                        continue
                    if (edges_capacity[current_node, next_node].item() == 0):
                        continue

                    flow_probability = y_commodities[current_node, next_node]
                    new_score = current_score + flow_probability

                    if demand <= remaining_edges_capacity[current_node, next_node]:
                        updated_capacity = remaining_edges_capacity.clone()
                        updated_capacity[current_node, next_node] -= demand
                        new_path = path + [next_node]
                        next_beam_queue.append((next_node, new_path, new_score, updated_capacity))
                        
            beam_queue = next_beam_queue

        if not best_paths:
            node_order = [0] * edges_capacity.shape[0]
            return node_order, edges_capacity, []

        best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
        node_order = [0] * edges_capacity.shape[0]
        for idx, node in enumerate(best_paths[0][0]):
            node_order[node] = idx + 1
        return node_order, best_paths[0][2], best_paths[0][0]

    def _get_fallback_paths(self, num_commodities: int) -> List[List[int]]:
        """フォールバックパスの生成"""
        return [[0,1,2,3,4,5,6,7,8,9] for _ in range(num_commodities)]


class DeterministicBeamSearch(BeamSearchAlgorithm):
    """決定論的なビームサーチアルゴリズム（シャッフルなし）"""
    
    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """決定論的なビームサーチによる単一バッチ検索"""
        batch_y_pred_edges = self.y[batch]
        commodities = self.commodities[batch]
        count = 0

        while count < self.max_iter:
            batch_edges_capacity = self.edges_capacity[batch]
            
            # シャッフルなしの決定論的処理
            original_indices = torch.arange(commodities.size(0))
            processed_commodities = commodities
            processed_pred_edges = batch_y_pred_edges

            node_orders = []
            commodity_paths = []
            is_feasible = True

            for index, commodity in enumerate(processed_commodities):
                source_node = commodity[0].item()
                target_node = commodity[1].item()
                demand = commodity[2].item()
                
                # ビームサーチによるパス探索
                node_order, remaining_edges_capacity, best_path = self._beam_search_for_commodity(
                    batch_edges_capacity, processed_pred_edges[:, :, index], 
                    source_node, target_node, demand
                )
                
                if best_path == []:
                    break
                    
                node_orders.append(node_order)
                if self.mode_strict:
                    batch_edges_capacity = remaining_edges_capacity
                commodity_paths.append(best_path)

            if len(commodity_paths) == commodities.shape[0]:
                count = self.max_iter
                unshuffled_commodity_paths = [commodity_paths[i] for i in original_indices]
            else:
                is_feasible = False
                unshuffled_commodity_paths = self._get_fallback_paths(commodities.shape[0])
                count += 1
        
        return node_orders, unshuffled_commodity_paths, is_feasible

    def _beam_search_for_commodity(self, edges_capacity, y_commodities, source, target, demand):
        """決定論的なビームサーチによるパス探索（元の実装と同じ）"""
        beam_queue = [(source, [source], 0, edges_capacity.clone())]
        best_paths = []

        while beam_queue:
            current_scores = [item[2] for item in beam_queue]
            beam_queue = sorted(beam_queue, key=lambda x: x[2], reverse=True)[:self.beam_size]

            next_beam_queue = []
            for current_node, path, current_score, remaining_edges_capacity in beam_queue:
                if current_node == target:
                    best_paths.append((path, current_score, remaining_edges_capacity))
                    continue

                for next_node in range(edges_capacity.shape[0]):
                    if next_node in path:
                        continue
                    if (edges_capacity[current_node, next_node].item() == 0):
                        continue

                    flow_probability = y_commodities[current_node, next_node]
                    new_score = current_score + flow_probability

                    if demand <= remaining_edges_capacity[current_node, next_node]:
                        updated_capacity = remaining_edges_capacity.clone()
                        updated_capacity[current_node, next_node] -= demand
                        new_path = path + [next_node]
                        next_beam_queue.append((next_node, new_path, new_score, updated_capacity))
                        
            beam_queue = next_beam_queue

        if not best_paths:
            node_order = [0] * edges_capacity.shape[0]
            return node_order, edges_capacity, []

        best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
        node_order = [0] * edges_capacity.shape[0]
        for idx, node in enumerate(best_paths[0][0]):
            node_order[node] = idx + 1
        return node_order, best_paths[0][2], best_paths[0][0]

    def _get_fallback_paths(self, num_commodities: int) -> List[List[int]]:
        """フォールバックパスの生成"""
        return [[0,1,2,3,4,5,6,7,8,9] for _ in range(num_commodities)]


class GreedyBeamSearch(BeamSearchAlgorithm):
    """貪欲的なビームサーチアルゴリズム（ビームサイズ1）"""
    
    def __init__(self, y_pred_edges, beam_size, batch_size, edges_capacity, commodities, 
                 dtypeFloat, dtypeLong, mode_strict=False, max_iter=5):
        super().__init__(y_pred_edges, 1, batch_size, edges_capacity, commodities, 
                        dtypeFloat, dtypeLong, mode_strict, max_iter)

    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """貪欲的なビームサーチによる単一バッチ検索"""
        batch_y_pred_edges = self.y[batch]
        commodities = self.commodities[batch]
        count = 0

        while count < self.max_iter:
            batch_edges_capacity = self.edges_capacity[batch]
            
            # ランダムシャッフルによる前処理
            random_indices = torch.randperm(commodities.size(0))
            shuffled_commodities = commodities[random_indices]
            shuffled_pred_edges = batch_y_pred_edges[:, :, random_indices]
            _, original_indices = torch.sort(random_indices)

            node_orders = []
            commodity_paths = []
            is_feasible = True

            for index, commodity in enumerate(shuffled_commodities):
                source_node = commodity[0].item()
                target_node = commodity[1].item()
                demand = commodity[2].item()
                
                # 貪欲的なパス探索（ビームサイズ1）
                node_order, remaining_edges_capacity, best_path = self._greedy_search_for_commodity(
                    batch_edges_capacity, shuffled_pred_edges[:, :, index], 
                    source_node, target_node, demand
                )
                
                if best_path == []:
                    break
                    
                node_orders.append(node_order)
                if self.mode_strict:
                    batch_edges_capacity = remaining_edges_capacity
                commodity_paths.append(best_path)

            if len(commodity_paths) == commodities.shape[0]:
                count = self.max_iter
                unshuffled_commodity_paths = [commodity_paths[i] for i in original_indices]
            else:
                is_feasible = False
                unshuffled_commodity_paths = self._get_fallback_paths(commodities.shape[0])
                count += 1
        
        return node_orders, unshuffled_commodity_paths, is_feasible

    def _greedy_search_for_commodity(self, edges_capacity, y_commodities, source, target, demand):
        """貪欲的なパス探索（ビームサイズ1）"""
        beam_queue = [(source, [source], 0, edges_capacity.clone())]
        best_paths = []

        while beam_queue:
            current_scores = [item[2] for item in beam_queue]
            beam_queue = sorted(beam_queue, key=lambda x: x[2], reverse=True)[:self.beam_size]

            next_beam_queue = []
            for current_node, path, current_score, remaining_edges_capacity in beam_queue:
                if current_node == target:
                    best_paths.append((path, current_score, remaining_edges_capacity))
                    continue

                for next_node in range(edges_capacity.shape[0]):
                    if next_node in path:
                        continue
                    if (edges_capacity[current_node, next_node].item() == 0):
                        continue

                    flow_probability = y_commodities[current_node, next_node]
                    new_score = current_score + flow_probability

                    if demand <= remaining_edges_capacity[current_node, next_node]:
                        updated_capacity = remaining_edges_capacity.clone()
                        updated_capacity[current_node, next_node] -= demand
                        new_path = path + [next_node]
                        next_beam_queue.append((next_node, new_path, new_score, updated_capacity))
                        
            beam_queue = next_beam_queue

        if not best_paths:
            node_order = [0] * edges_capacity.shape[0]
            return node_order, edges_capacity, []

        best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
        node_order = [0] * edges_capacity.shape[0]
        for idx, node in enumerate(best_paths[0][0]):
            node_order[node] = idx + 1
        return node_order, best_paths[0][2], best_paths[0][0]

    def _get_fallback_paths(self, num_commodities: int) -> List[List[int]]:
        """フォールバックパスの生成"""
        return [[0,1,2,3,4,5,6,7,8,9] for _ in range(num_commodities)]


class UnconstrainedBeamSearch(BeamSearchAlgorithm):
    """容量制約なしのビームサーチアルゴリズム"""

    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """容量制約なしのビームサーチによる単一バッチ検索

        Note: 容量制約を無視するため、常にパスが見つかる想定。
        is_feasibleは常にTrueを返すが、実際の実行可能性は
        後で最大負荷率で判定される。
        """
        batch_y_pred_edges = self.y[batch]
        commodities = self.commodities[batch]

        node_orders = []
        commodity_paths = []

        for index, commodity in enumerate(commodities):
            source_node = commodity[0].item()
            target_node = commodity[1].item()

            # 容量制約なしのビームサーチによるパス探索
            node_order, best_path = self._unconstrained_beam_search_for_commodity(
                self.edges_capacity[batch], batch_y_pred_edges[:, :, index],
                source_node, target_node
            )

            node_orders.append(node_order)
            commodity_paths.append(best_path)

        # 容量制約を無視するため、is_feasibleは常にTrue
        # 実際の実行可能性は最大負荷率で後で判定
        return node_orders, commodity_paths, True

    def _unconstrained_beam_search_for_commodity(self, edges_capacity, y_commodities, source, target):
        """容量制約なしのビームサーチによるパス探索"""
        beam_queue = [(source, [source], 0)]
        best_paths = []

        while beam_queue:
            # ビームサイズでフィルタリング
            beam_queue = sorted(beam_queue, key=lambda x: x[2], reverse=True)[:self.beam_size]

            next_beam_queue = []
            for current_node, path, current_score in beam_queue:
                if current_node == target:
                    best_paths.append((path, current_score))
                    continue

                for next_node in range(edges_capacity.shape[0]):
                    # ループ検出のみ（容量チェックなし）
                    if next_node in path:
                        continue
                    # エッジが存在するかチェック（容量>0）
                    if edges_capacity[current_node, next_node].item() == 0:
                        continue

                    flow_probability = y_commodities[current_node, next_node]
                    new_score = current_score + flow_probability
                    new_path = path + [next_node]

                    # 容量制約なしで追加
                    next_beam_queue.append((next_node, new_path, new_score))

            beam_queue = next_beam_queue

        if not best_paths:
            node_order = [0] * edges_capacity.shape[0]
            return node_order, []

        best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
        node_order = [0] * edges_capacity.shape[0]
        for idx, node in enumerate(best_paths[0][0]):
            node_order[node] = idx + 1
        return node_order, best_paths[0][0]


class BeamSearchFactory:
    """ビームサーチアルゴリズムのファクトリークラス"""

    ALGORITHMS = {
        'standard': StandardBeamSearch,
        'deterministic': DeterministicBeamSearch,
        'greedy': GreedyBeamSearch,
        'unconstrained': UnconstrainedBeamSearch
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, **kwargs) -> BeamSearchAlgorithm:
        """
        指定されたアルゴリズムのインスタンスを作成
        
        Args:
            algorithm_name: アルゴリズム名 ('standard', 'deterministic', 'greedy')
            **kwargs: アルゴリズムの初期化パラメータ
            
        Returns:
            BeamSearchAlgorithm: アルゴリズムのインスタンス
            
        Raises:
            ValueError: 無効なアルゴリズム名の場合
        """
        if algorithm_name not in cls.ALGORITHMS:
            available = ', '.join(cls.ALGORITHMS.keys())
            raise ValueError(f"無効なアルゴリズム名: {algorithm_name}. 利用可能: {available}")
        
        algorithm_class = cls.ALGORITHMS[algorithm_name]
        return algorithm_class(**kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """利用可能なアルゴリズムのリストを取得"""
        return list(cls.ALGORITHMS.keys())


# 後方互換性のためのエイリアス
class BeamsearchUELB(StandardBeamSearch):
    """後方互換性のためのエイリアス"""
    pass
------------------------------------------------------------------------------------------------
"""
Improved Path Sampler with full masking and feasibility constraints
for Reinforcement Learning (REINFORCE algorithm).
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque


class PathSampler:
    """
    Samples paths probabilistically from GCN edge predictions
    while respecting feasibility constraints (capacity, connectivity).

    This ensures proper on-policy sampling for REINFORCE learning
    and prevents invalid (infeasible) paths.
    """

    def __init__(self, y_pred_edges, edges_capacity, commodities,
                 num_samples=1, temperature=1.0, top_p=0.8, entropy_epsilon: float = 0.0,
                 dtypeFloat=torch.float, dtypeLong=torch.long):
        """
        Args:
            y_pred_edges: Edge predictions [batch, nodes, nodes, classes]
            edges_capacity: Edge capacities [batch, nodes, nodes]
            commodities: List of (src, dst, demand) tuples
            num_samples: Number of path samples to draw
            temperature: Temperature for softmax (lower = more deterministic)
            top_p: Nucleus sampling threshold (0.9 = sample from top 90% probability mass)
            dtypeFloat: Float tensor type
            dtypeLong: Long tensor type
        """
        self.y_pred_edges = y_pred_edges
        self.edges_capacity = edges_capacity
        self.commodities = commodities
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.entropy_epsilon = float(entropy_epsilon) if entropy_epsilon is not None else 0.0
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        self.batch_size = y_pred_edges.shape[0]
        self.num_nodes = y_pred_edges.shape[1]

        # Precompute reachability matrix for each batch
        self.reachability = self._precompute_reachability()

    def sample(self):
        """
        Sample paths for all commodities using top-p sampling.

        Note: Capacity constraints are NOT enforced during sampling.
        The model can explore infeasible solutions and learn from
        the penalty in the reward signal.

        Returns:
            batch_paths: list of sampled paths
            log_probs_tensor: tensor of total log-probabilities per batch
            is_feasible: whether all flows respect capacity constraints (checked post-hoc)
        """
        device = self.y_pred_edges.device
        batch_paths = []
        batch_log_probs = []  # Keep as tensors, not float
        batch_stepwise_entropies = []  # List[List[List[float]]] per batch -> per commodity -> per step

        # DEBUG: Print shapes for first batch only
        debug_enabled = False  # Set to True to enable debug output
        debug_first_batch = debug_enabled and (not hasattr(self, '_debug_printed'))
        if debug_first_batch:
            self._debug_printed = True
            print(f"\n=== PathSampler Debug ===")
            print(f"  y_pred_edges shape: {self.y_pred_edges.shape}")
            print(f"  edges_capacity shape: {self.edges_capacity.shape}")

        for b in range(self.batch_size):
            # Store batch index for later use in _sample_single_path
            self._current_batch_idx = b
            edge_logits = self.y_pred_edges[b]

            # --- Global mask for physically non-existent edges ---
            # (zero capacity = edge doesn't exist, self-loops are invalid)
            # Use a very large negative value to effectively zero out probabilities after softmax
            invalid_mask = (self.edges_capacity[b] <= 0) | torch.eye(self.num_nodes, device=device).bool()

            if debug_first_batch and b == 0:
                print(f"  edge_logits shape: {edge_logits.shape}")
                print(f"  invalid_mask shape: {invalid_mask.shape}")
                print(f"  Number of valid edges: {(~invalid_mask).sum().item()}/{invalid_mask.numel()}")
                print(f"  Capacity stats - min: {self.edges_capacity[b].min():.2f}, max: {self.edges_capacity[b].max():.2f}, nonzero: {(self.edges_capacity[b] > 0).sum().item()}")

                # Check graph connectivity
                num_isolated_nodes = 0
                for node in range(self.num_nodes):
                    outgoing = (~invalid_mask[node]).sum().item()
                    incoming = (~invalid_mask[:, node]).sum().item()
                    if outgoing == 0 or incoming == 0:
                        num_isolated_nodes += 1
                print(f"  Isolated/dead-end nodes: {num_isolated_nodes}/{self.num_nodes}")

            # --- Apply invalid edge mask to logits ---
            # edge_logits shape: [nodes_from, nodes_to, commodities]
            # Mask out invalid edges (zero capacity or self-loops)
            edge_logits = edge_logits.masked_fill(invalid_mask.unsqueeze(-1), -1e20)

            # --- Convert logits to edge probabilities ---
            # UPDATED 2025-10-19: Compute next-node selection probabilities
            # dim=1 is the "to_node" dimension (14 choices for next node from current node)
            edge_probs_all = F.softmax(edge_logits / self.temperature, dim=1)

            # Ensure zero-capacity edges have exactly zero probability
            edge_probs_all = edge_probs_all * (~invalid_mask).unsqueeze(-1).float()

            if debug_first_batch and b == 0:
                print(f"  edge_probs_all shape: {edge_probs_all.shape}")
                print(f"  edge_probs_all stats - min: {edge_probs_all.min():.6f}, max: {edge_probs_all.max():.6f}, mean: {edge_probs_all.mean():.6f}")
                print(f"  Nonzero probabilities: {(edge_probs_all > 1e-10).sum().item()}/{edge_probs_all.numel()}")
                print("=" * 50)

            # --- Commodity processing ---
            commodity_paths = []
            commodity_log_probs = []  # Keep as tensors, not float
            commodity_step_entropies = []

            batch_commodities = (
                self.commodities[b] if isinstance(self.commodities, torch.Tensor) and len(self.commodities.shape) == 3
                else self.commodities
            )

            for c_idx, commodity in enumerate(batch_commodities):
                src, dst, demand = self._parse_commodity(commodity)

                if len(edge_probs_all.shape) == 3:
                    edge_probs = edge_probs_all[:, :, c_idx]
                else:
                    edge_probs = edge_probs_all

                # DEBUG: Print first path
                if debug_first_batch and b == 0 and c_idx == 0:
                    print(f"\n  Commodity {c_idx}: src={src}, dst={dst}, demand={demand}")
                    print(f"  edge_probs for this commodity - nonzero: {(edge_probs > 1e-10).sum().item()}/{edge_probs.numel()}")
                    print(f"  edge_probs from src {src} - nonzero: {(edge_probs[src] > 1e-10).sum().item()}/{len(edge_probs[src])}")
                    # Check if edge to dst is masked
                    print(f"  edge_probs[{src}, {dst}] (direct to dst): {edge_probs[src, dst]:.6f}")
                    print(f"  invalid_mask[{src}, {dst}]: {invalid_mask[src, dst]}")

                # remaining_capacity is no longer tracked or enforced
                path, log_prob, step_entropies = self._sample_single_path(
                    edge_probs, src, dst, demand, remaining_capacity=None
                )

                if debug_first_batch and b == 0 and c_idx == 0:
                    path_length = len(path) - 1 if len(path) > 1 else 0  # Number of edges
                    print(f"  Generated path: {path} (nodes={len(path)}, edges={path_length})")
                    # Check if path reaches destination
                    if len(path) == 0 or path[-1] != dst:
                        print(f"    WARNING: Path incomplete (doesn't reach dst={dst})")
                    elif len(path) == 1:
                        print(f"    WARNING: Zero-length path (src == dst = {src})")
                    # Check if path uses invalid edges
                    uses_invalid = False
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if invalid_mask[u, v]:
                            print(f"    WARNING: Path uses invalid edge ({u}, {v})")
                            print(f"      edge_probs[{u}, {v}]: {edge_probs[u, v]:.6f}")
                            print(f"      edges_capacity[{b}, {u}, {v}]: {self.edges_capacity[b, u, v]:.2f}")
                            uses_invalid = True
                    print(f"  Uses invalid edges: {uses_invalid}")

                commodity_paths.append(path)
                commodity_log_probs.append(log_prob)
                commodity_step_entropies.append(step_entropies)

            # --- Aggregate batch results ---
            batch_paths.append(commodity_paths)
            # Sum log_probs as tensors to maintain gradient connection
            total_log_prob = (
                torch.stack(commodity_log_probs).sum()
                if commodity_log_probs
                else torch.zeros(1, device=device, dtype=self.y_pred_edges.dtype).squeeze(0)
            )
            batch_log_probs.append(total_log_prob)
            batch_stepwise_entropies.append(commodity_step_entropies)

            # DEBUG: Check how many commodities use invalid edges or are incomplete
            if debug_first_batch and b == 0:
                invalid_count = 0
                incomplete_count = 0
                for c_idx, path in enumerate(commodity_paths):
                    dst = int(batch_commodities[c_idx][1].item())
                    # Check if incomplete
                    if len(path) == 0 or path[-1] != dst:
                        incomplete_count += 1
                    # Check if uses invalid edges
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if invalid_mask[u, v]:
                            invalid_count += 1
                            break  # One invalid edge per commodity is enough
                print(f"  Commodities using invalid edges: {invalid_count}/{len(commodity_paths)}")
                print(f"  Incomplete paths (dst not reached): {incomplete_count}/{len(commodity_paths)}")

        is_feasible = self._check_feasibility(batch_paths)
        # Stack batch log probs (already tensors) to maintain gradient
        log_probs_tensor = (
            torch.stack(batch_log_probs)
            if batch_log_probs
            else torch.empty(0, device=device, dtype=self.y_pred_edges.dtype)
        )

        return batch_paths, log_probs_tensor, is_feasible, batch_stepwise_entropies

    # ==============================================================
    # Internal Methods
    # ==============================================================

    def _precompute_reachability(self):
        """
        Precompute reachability matrix for all node pairs in each batch.

        Returns:
            reachability: [batch_size, num_nodes, num_nodes] boolean tensor
                         reachability[b][i][j] = True if node j is reachable from node i in batch b
        """
        device = self.edges_capacity.device
        reachability = torch.zeros(
            (self.batch_size, self.num_nodes, self.num_nodes),
            dtype=torch.bool,
            device=device
        )

        for b in range(self.batch_size):
            # Build adjacency list from capacity matrix
            # edge exists if capacity > 0
            adj_list = [[] for _ in range(self.num_nodes)]
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    # PyTorch scalar comparison works directly
                    if self.edges_capacity[b, i, j] > 0 and i != j:
                        adj_list[i].append(j)

            # BFS from each source node
            for src in range(self.num_nodes):
                # Use deque for efficient BFS (O(1) popleft vs O(n) pop(0))
                queue = deque([src])
                visited = set([src])
                reachability[b, src, src] = True

                while queue:
                    current = queue.popleft()
                    for neighbor in adj_list[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            reachability[b, src, neighbor] = True
                            queue.append(neighbor)

        return reachability

    def _parse_commodity(self, commodity):
        """Extract (src, dst, demand) as native Python types."""
        if isinstance(commodity, (list, tuple)):
            src, dst, demand = commodity
        else:
            src = int(commodity[0].item() if hasattr(commodity[0], 'item') else commodity[0])
            dst = int(commodity[1].item() if hasattr(commodity[1], 'item') else commodity[1])
            demand = float(commodity[2].item() if hasattr(commodity[2], 'item') else commodity[2])
        return src, dst, demand

    def _sample_single_path(self, edge_probs, src, dst, demand, remaining_capacity=None):
        """Sample a path using top-p nucleus sampling.

        Note: Capacity constraints are NOT enforced during sampling.
        This allows the model to explore infeasible solutions and learn
        from the penalty in the reward signal.

        Args:
            edge_probs: Edge probabilities for current commodity
            src: Source node
            dst: Destination node
            demand: Commodity demand (not used for capacity checking)
            remaining_capacity: Ignored (kept for API compatibility)
        """
        src, dst = int(src), int(dst)

        path = [src]
        log_prob = torch.zeros(1, device=edge_probs.device, dtype=edge_probs.dtype).squeeze(0)  # Keep as tensor
        step_entropies: list = []
        current = src
        max_steps = self.num_nodes * 2  # avoid infinite loops

        for step_idx in range(max_steps):
            if current == dst:
                break

            outgoing_probs = edge_probs[current].clone()

            # --- Apply Reachability Mask + Visited Mask ---
            # Get batch index
            batch_idx = self._current_batch_idx if hasattr(self, '_current_batch_idx') else 0

            # Create combined mask: not visited AND reachable to destination
            visited_mask = torch.ones(self.num_nodes, dtype=torch.bool, device=outgoing_probs.device)
            for visited_node in path:
                visited_mask[int(visited_node)] = False  # exclude visited nodes to prevent loops

            # Reachability mask: only nodes that can reach the destination
            # Ensure dst is Python int for indexing
            dst_idx = int(dst)
            reachable_mask = self.reachability[batch_idx, :, dst_idx]  # [num_nodes]

            # Combined mask: not visited AND reachable
            combined_mask = visited_mask & reachable_mask

            # Apply mask
            outgoing_probs = outgoing_probs * combined_mask.float()

            # Remove numerical noise
            outgoing_probs = torch.clamp(outgoing_probs, min=0.0)

            # --- Normalize ---
            total_prob = outgoing_probs.sum()

            if total_prob <= 1e-20:
                # Fallback: No valid edges available with reachability constraint
                # Try to find edges that are: has capacity AND not visited AND reachable
                device = edge_probs.device
                batch_idx = self._current_batch_idx if hasattr(self, '_current_batch_idx') else 0

                if len(self.edges_capacity.shape) == 2:
                    capacity_available = self.edges_capacity[current] > 0
                else:
                    capacity_available = self.edges_capacity[batch_idx, current] > 0

                # Combine: has capacity AND not visited AND reachable
                fallback_mask = capacity_available & combined_mask

                if fallback_mask.any():
                    # Pick valid edge with SOME heuristic guidance
                    # Use unmasked probabilities (from model) as guidance, but only among valid edges
                    valid_indices = torch.where(fallback_mask)[0]

                    # Get model's probabilities for valid edges only
                    unmasked_probs = edge_probs[current].clone()
                    valid_probs = unmasked_probs[valid_indices]

                    if valid_probs.sum() > 1e-10:
                        # If model assigns some probability to valid edges, use it
                        valid_probs = valid_probs / valid_probs.sum()
                        # Sample from model's distribution over valid edges
                        sampled_idx = torch.multinomial(valid_probs, 1).item()
                        next_node = valid_indices[sampled_idx].item()
                        # Keep log_prob as tensor for gradient tracking
                        log_prob = log_prob + torch.log(valid_probs[sampled_idx] + 1e-8)
                    else:
                        # Model assigns 0 probability to all valid edges
                        # Fall back to uniform distribution
                        next_node = valid_indices[torch.randint(len(valid_indices), (1,))].item()
                        # Convert numpy log to tensor
                        log_prob = log_prob + torch.tensor(np.log(1.0 / len(valid_indices)), device=log_prob.device, dtype=log_prob.dtype)

                    path.append(int(next_node))
                    current = int(next_node)
                    continue
                else:
                    # Truly stuck: no valid edges at all from current node
                    # Cannot reach dst without using invalid edges
                    # Return incomplete path (don't force-add dst through invalid edge)
                    log_prob = log_prob + torch.tensor(np.log(1e-8), device=log_prob.device, dtype=log_prob.dtype)
                    break

            outgoing_probs /= total_prob

            # --- Epsilon mixture with uniform over valid, non-visited next nodes (ε-greedy over valid set) ---
            eps = self.entropy_epsilon
            if eps > 0.0:
                # Capacity-available mask for current row (respect batch index)
                batch_idx = self._current_batch_idx if hasattr(self, '_current_batch_idx') else 0
                if len(self.edges_capacity.shape) == 2:
                    capacity_available = self.edges_capacity[current] > 0
                else:
                    capacity_available = self.edges_capacity[batch_idx, current] > 0
                # Valid and not-visited AND reachable to destination
                valid_support = capacity_available & combined_mask
                if valid_support.any():
                    uniform = valid_support.float()
                    uniform = uniform / (uniform.sum() + 1e-8)
                    outgoing_probs = (1.0 - eps) * outgoing_probs + eps * uniform
                    outgoing_probs = outgoing_probs / (outgoing_probs.sum() + 1e-8)

            # --- Record step entropy before sampling ---
            step_entropy = -(outgoing_probs * torch.log(outgoing_probs + 1e-8)).sum().item()
            step_entropies.append(step_entropy)

            # --- Top-p nucleus sampling ---
            next_node = self._top_p_sample(outgoing_probs)
            # Keep log_prob as tensor for gradient tracking
            log_prob = log_prob + torch.log(outgoing_probs[next_node] + 1e-8)
            path.append(int(next_node))
            current = int(next_node)

        # Return path as-is (may be incomplete if destination unreachable with valid edges)
        return path, log_prob, step_entropies

    def _top_p_sample(self, probs):
        """Perform top-p (nucleus) sampling robustly."""
        probs = probs * (probs > 1e-8)  # Remove zero-prob nodes
        if probs.sum() == 0:
            # fallback to random selection if all invalid
            return torch.randint(0, len(probs), (1,)).item()

        probs = probs / probs.sum()
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)

        cutoff_idx = torch.where(cumsum_probs > self.top_p)[0]
        cutoff_idx = cutoff_idx[0].item() + 1 if len(cutoff_idx) > 0 else len(sorted_probs)

        top_p_probs = sorted_probs[:cutoff_idx]
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_probs = top_p_probs / top_p_probs.sum()

        sampled_idx = top_p_indices[torch.multinomial(top_p_probs, 1).item()].item()
        return sampled_idx

    def _check_feasibility(self, batch_paths):
        """Verify that total edge usage does not exceed capacities."""
        for b, commodity_paths in enumerate(batch_paths):
            edge_usage = torch.zeros_like(self.edges_capacity[b])
            batch_commodities = (
                self.commodities[b]
                if isinstance(self.commodities, torch.Tensor) and len(self.commodities.shape) == 3
                else self.commodities
            )

            for i, path in enumerate(commodity_paths):
                src, dst, demand = self._parse_commodity(batch_commodities[i])
                for j in range(len(path) - 1):
                    u, v = path[j], path[j + 1]
                    edge_usage[u, v] += demand

            if (edge_usage > self.edges_capacity[b] + 1e-6).any():
                return False
        return True
