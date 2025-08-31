#!/usr/bin/env python3
"""
Reinforcement Learning Trainer for Graph Convolutional Network UELB
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict, Any
import csv
import os
import time
import datetime
import hashlib
import json
from pathlib import Path

from .rl_environment import MinMaxLoadKSPsEnv
from ..data_management.dataset_reader import DatasetReader
from ..train.metrics import MetricsLogger
from ..data_management.exact_solution import SolveExactSolution


class DQNModel(nn.Module):
    """
    Deep Q-Network model for reinforcement learning
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super(DQNModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 32, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class RLTrainer:
    """
    強化学習トレーナークラス
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        トレーナーの初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # モデル保存設定
        self.models_dir = Path(config.get('models_dir', './saved_models'))
        self.models_dir.mkdir(exist_ok=True)
        
        # データセットリーダーの初期化
        num_train_data = config.get('num_train_data', 100)
        batch_size = 1  # RL用には1つずつ処理
        self.train_dataset = DatasetReader(num_train_data, batch_size, 'train')
        self.test_dataset = DatasetReader(config.get('num_test_data', 20), batch_size, 'test')
        
        # 環境の初期化（既存データを使用）
        self.env = MinMaxLoadKSPsEnv(config)
        
        # ニューラルネットワークの初期化
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        hidden_dims = config.get('hidden_dims', [32, 32, 32])
        
        # モデルとオプティマイザーの初期化（保存済みモデルの読み込みを考慮）
        self.model, self.optimizer = self._instantiate_model(input_dim, output_dim, hidden_dims)
        
        # 学習パラメータ
        self.epsilon = config.get('epsilon', 0.8)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.gamma = config.get('gamma', 0.85)
        self.episodes = config.get('episodes', 1000)
        
        # 結果保存とメトリクス
        self.results_dir = Path('./results')
        self.results_dir.mkdir(exist_ok=True)
        self.metrics_logger = MetricsLogger(save_dir="logs")
        
        # 厳密解計算用の設定
        self.solver_type = config.get('solver_type', 'pulp')
        self.config = config
    
    def _instantiate_model(self, input_dim: int, output_dim: int, hidden_dims: List[int]) -> Tuple[nn.Module, optim.Optimizer]:
        """
        モデルとオプティマイザーを初期化（保存済みモデルの読み込みを考慮）
        
        Args:
            input_dim: 入力次元
            output_dim: 出力次元
            hidden_dims: 隠れ層の次元リスト
            
        Returns:
            (model, optimizer): モデルとオプティマイザーのタプル
        """
        # 既存の保存済みモデルをチェック
        if self.config.get('load_saved_model', False):
            loaded_model, loaded_optimizer = self._try_load_saved_model(input_dim, output_dim, hidden_dims)
            if loaded_model is not None:
                return loaded_model, loaded_optimizer
        
        # 新しいモデルを作成
        model = DQNModel(input_dim, output_dim, hidden_dims).to(self.device)
        
        # オプティマイザーの設定
        learning_rate = self.config.get('learning_rate', 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"Created new RL model with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Model architecture: {input_dim} -> {hidden_dims} -> {output_dim}")
        print(f"Device: {self.device}")
        
        return model, optimizer
    
    def _get_config_hash(self) -> str:
        """設定のハッシュを生成してモデル識別に使用"""
        config_for_hash = {
            'hidden_dims': self.config.get('hidden_dims', [32, 32, 32]),
            'learning_rate': self.config.get('learning_rate', 0.0001),
            'gamma': self.config.get('gamma', 0.85),
            'epsilon': self.config.get('epsilon', 0.8),
            'epsilon_decay': self.config.get('epsilon_decay', 0.995),
            'n_action': self.config.get('n_action', 20),
            'obs_low': self.config.get('obs_low', -20),
            'obs_high': self.config.get('obs_high', 20),
        }
        config_str = json.dumps(config_for_hash, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_model_filename(self, episode: Optional[int] = None) -> str:
        """モデルファイル名を生成"""
        config_hash = self._get_config_hash()
        if episode is not None:
            return f"rl_model_{config_hash}_episode_{episode}.pt"
        else:
            return f"rl_model_{config_hash}_latest.pt"
    
    def _try_load_saved_model(self, input_dim: int, output_dim: int, hidden_dims: List[int]) -> Tuple[Optional[nn.Module], Optional[optim.Optimizer]]:
        """保存済みモデルの読み込みを試行"""
        if not self.models_dir.exists():
            return None, None
        
        # 特定のエピソードが指定されている場合はそれを優先
        load_episode = self.config.get('load_model_episode', None)
        if load_episode is not None:
            model_filename = self._get_model_filename(episode=load_episode)
        else:
            model_filename = self._get_model_filename()
        
        model_path = self.models_dir / model_filename
        
        if not model_path.exists():
            print(f"No saved RL model found at {model_path}")
            self._show_available_models()
            return None, None
        
        try:
            print(f"Loading saved RL model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # モデル構造の検証
            expected_config = checkpoint.get('config_hash')
            current_config = self._get_config_hash()
            if expected_config != current_config:
                print(f"Config mismatch: expected {expected_config}, got {current_config}")
                return None, None
            
            # モデルとオプティマイザーの復元
            model = DQNModel(input_dim, output_dim, hidden_dims).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            learning_rate = self.config.get('learning_rate', 0.0001)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 追加パラメータの復元
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            
            print(f"Successfully loaded RL model from episode {checkpoint.get('episode', 'unknown')}")
            print(f"Previous training loss: {checkpoint.get('loss', 'unknown')}")
            print(f"Previous epsilon: {checkpoint.get('epsilon', 'unknown')}")
            
            return model, optimizer
            
        except Exception as e:
            print(f"Failed to load RL model: {e}")
            return None, None
    
    def _show_available_models(self):
        """利用可能なモデルファイルを表示"""
        if not self.models_dir.exists():
            print("No saved models directory found.")
            return
        
        config_hash = self._get_config_hash()
        matching_models = []
        
        for model_file in self.models_dir.glob("rl_model_*.pt"):
            if f"rl_model_{config_hash}_" in model_file.name:
                matching_models.append(model_file.name)
        
        if matching_models:
            print("Available RL models for current configuration:")
            for model in sorted(matching_models):
                model_path = self.models_dir / model
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    episode = checkpoint.get('episode', 'unknown')
                    loss = checkpoint.get('loss', 'unknown')
                    epsilon = checkpoint.get('epsilon', 'unknown')
                    if isinstance(loss, (int, float)):
                        print(f"  - {model} (episode: {episode}, loss: {loss:.4f}, epsilon: {epsilon})")
                    else:
                        print(f"  - {model} (episode: {episode}, loss: {loss}, epsilon: {epsilon})")
                except:
                    print(f"  - {model} (info unavailable)")
        else:
            print(f"No RL models found for current configuration (hash: {config_hash})")
            # 他の設定のモデルがある場合は表示
            all_models = list(self.models_dir.glob("rl_model_*.pt"))
            if all_models:
                print("Available RL models for other configurations:")
                for model in sorted(all_models)[:5]:
                    print(f"  - {model.name}")
                if len(all_models) > 5:
                    print(f"  ... and {len(all_models) - 5} more")
        
    def choose_action(self, state: np.ndarray) -> int:
        """
        ニューラルネットワークの予測値に基づく行動選択
        
        Args:
            state: 現在の状態
            
        Returns:
            選択された行動
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            
            # 観測変数が-100.0の場合は、その行動を選択しない
            filtered_q_values = torch.where(
                torch.FloatTensor(state).to(self.device) == -100.0,
                torch.FloatTensor([-float('inf')]).to(self.device),
                q_values.squeeze(0)
            )
            
            action = torch.argmax(filtered_q_values).item()
            return action
    
    def train_episode(self, episode: int) -> Tuple[float, float]:
        """
        1エピソードの学習
        
        Args:
            episode: エピソード番号
            
        Returns:
            (total_reward, loss)
        """
        total_reward = 0.0
        total_loss = 0.0
        step_count = 0
        
        state = self.env.reset()
        state = np.reshape(state, [1, self.env.observation_space.shape[0]])
        done = False
        
        while not done:
            # ε-greedy行動選択
            if np.random.rand() <= self.epsilon:
                # 探索: ランダム行動
                action = np.random.randint(self.env.action_space.n)
            else:
                # 活用: モデルによる予測
                action = self.choose_action(state[0])
            
            # 行動実行
            next_state, reward, done, _, max_load = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
            total_reward += reward
            
            # Q学習の更新
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            
            target = reward
            if not done:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor)
                    target = reward + self.gamma * torch.max(next_q_values).item()
            
            # 現在の状態でのQ値予測
            current_q_values = self.model(state_tensor)
            target_q_values = current_q_values.clone()
            target_q_values[0][action] = target
            
            # 損失計算と逆伝播
            loss = nn.MSELoss()(current_q_values, target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            step_count += 1
            
            # 状態の更新
            state = next_state
            
            # εの更新
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        
        avg_loss = total_loss / step_count if step_count > 0 else 0.0
        return total_reward, avg_loss
    
    def train(self) -> None:
        """
        学習メインループ
        """
        print(f"Starting RL training for {self.episodes} episodes")
        
        # 学習履歴保存用
        training_history = []
        
        for episode in range(self.episodes):
            start_time = time.time()
            total_reward, loss = self.train_episode(episode)
            episode_time = time.time() - start_time
            
            # メトリクス記録（損失値、報酬改善率、時間）
            reward_improvement = abs(total_reward)  # 負の報酬なので絶対値が改善率
            self.metrics_logger.log_train_metrics(loss, reward_improvement, episode_time)
            
            # 履歴の保存
            training_history.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'loss': loss,
                'epsilon': self.epsilon,
                'time': episode_time
            })
            
            # 進捗表示（10エピソード刻み）
            if (episode + 1) % 10 == 0 or episode == 0:
                print(f"Episode {episode + 1}/{self.episodes}, "
                      f"Total Reward: {total_reward:.4f}, "
                      f"Loss: {loss:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"Time: {episode_time:.2f}s")
            
            # 定期的なモデル保存
            save_every = self.config.get('save_every_n_episodes', 0)
            if save_every > 0 and (episode + 1) % save_every == 0:
                self.save_model(episode + 1, loss, save_latest=False)
                print(f"Checkpoint saved at episode {episode + 1}")
        
        # 学習履歴の保存
        self._save_training_history(training_history)
        
        # モデルの保存
        final_loss = training_history[-1]['loss'] if training_history else 0.0
        self.save_model(self.episodes, final_loss, save_latest=True)
        
        print("Training completed!")
    
    def test(self, test_episodes: int = 100) -> None:
        """
        テストの実行
        
        Args:
            test_episodes: テストエピソード数
        """
        print(f"Starting RL testing for {test_episodes} episodes")
        
        test_results = []
        total_test_time = 0.0
        
        for episode in range(test_episodes):
            start_time = time.time()
            state = self.env.reset('test')  # テストモードを指定
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            done = False
            total_reward = 0.0
            steps = 0
            
            while not done:
                action = self.choose_action(state[0])
                next_state, reward, done, _, max_load = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
                
                total_reward += reward
                steps += 1
                state = next_state
            
            episode_time = time.time() - start_time
            total_test_time += episode_time
            
            # 実際にRLが使用したデータのインデックスを取得
            current_data_idx = getattr(self.env, 'current_used_data_idx', self.env.data_idx - 1)
            gt_load_factor = self._get_actual_load_factor(current_data_idx)
            
            # RLが実際に使用したファイルパスをデバッグ出力
            graph_file = f"./data/test_data/graph_file/{current_data_idx-(current_data_idx%10)}/graph_{current_data_idx}.gml"
            commodity_file = f"./data/test_data/commodity_file/{current_data_idx-(current_data_idx%10)}/commodity_data_{current_data_idx}.csv"
            # print(f"    RL actually used files: {graph_file}, {commodity_file}")
            
            # GCNと同じ計算方法: gt_load_factor / predicted_load_factor * 100
            if max_load > 0:
                approximation_rate = gt_load_factor / max_load * 100
            else:
                approximation_rate = 0.0
            
            # デバッグ情報
            print(f"  Debug: data_idx={current_data_idx}, gt_load={gt_load_factor:.6f}, rl_load={max_load:.6f}, approx_rate={approximation_rate:.2f}%")
            
            # Approximation Rateが100%を明確に超えた場合のみ異常とみなして強制終了
            # 浮動小数点の精度を考慮して100.01%以上を異常とする
            if approximation_rate > 100.01:
                print(f"\n❌ CRITICAL ERROR: Approximation Rate exceeded 100% ({approximation_rate:.2f}%)")
                print(f"   This indicates a serious data inconsistency or calculation bug.")
                print(f"   Episode: {episode + 1}/{test_episodes}")
                print(f"   Data Index: {current_data_idx}")
                print(f"   GT Load Factor: {gt_load_factor:.6f}")
                print(f"   RL Max Load: {max_load:.6f}")
                print(f"   Files used: {graph_file}")
                print(f"                {commodity_file}")
                print(f"\nProgram terminated to prevent invalid results.")
                exit(1)
            
            self.metrics_logger.log_test_metrics(approximation_rate, episode_time)
            
            test_results.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'max_load': max_load,
                'steps': steps,
                'time': episode_time
            })
            
            # テスト進捗表示（10エピソード刻み）
            if (episode + 1) % 10 == 0 or episode == 0:
                print(f"Test Episode {episode + 1}/{test_episodes}, "
                      f"Total Reward: {total_reward:.4f}, "
                      f"Max Load: {max_load:.4f}, "
                      f"Steps: {steps}, "
                      f"Time: {episode_time:.2f}s")
        
        # テスト結果の保存
        self._save_test_results(test_results)
        
        # 統計の表示（GCNと同じスタイル）
        avg_reward = np.mean([r['total_reward'] for r in test_results])
        avg_max_load = np.mean([r['max_load'] for r in test_results])
        avg_steps = np.mean([r['steps'] for r in test_results])
        avg_time = total_test_time / test_episodes
        
        print(f"\n" + "="*50)
        print(f"RL TEST RESULTS SUMMARY")
        print(f"="*50)
        print(f"Average Total Reward: {avg_reward:.6f}")
        print(f"Average Max Load Factor: {avg_max_load:.6f}")
        print(f"Average Steps per Episode: {avg_steps:.2f}")
        print(f"Average Time per Episode: {avg_time:.4f}s")
        print(f"Total Test Time: {total_test_time:.2f}s")
        print(f"="*50)
    
    def _save_training_history(self, history: List[Dict]) -> None:
        """学習履歴の保存"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f'rl_training_history_{timestamp}.csv'
        
        with open(filename, 'w', newline='') as f:
            if history:
                writer = csv.DictWriter(f, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
    
    def _save_test_results(self, results: List[Dict]) -> None:
        """テスト結果の保存"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f'rl_test_results_{timestamp}.csv'
        
        with open(filename, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    
    def save_model(self, episode: int, loss: float, save_latest: bool = True) -> None:
        """
        モデルを保存（GCN同様の機能を持つ拡張版）
        
        Args:
            episode: 現在のエピソード
            loss: 現在の損失
            save_latest: 最新モデルとして保存するかどうか
        """
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'epsilon': self.epsilon,
            'config_hash': self._get_config_hash(),
            'config': {
                'hidden_dims': self.config.get('hidden_dims', [32, 32, 32]),
                'learning_rate': self.config.get('learning_rate', 0.0001),
                'gamma': self.config.get('gamma', 0.85),
                'epsilon_initial': self.config.get('epsilon', 0.8),
                'epsilon_decay': self.config.get('epsilon_decay', 0.995),
                'n_action': self.config.get('n_action', 20),
                'obs_low': self.config.get('obs_low', -20),
                'obs_high': self.config.get('obs_high', 20),
            }
        }
        
        # エピソード別保存
        if self.config.get('save_every_episode', False):
            episode_filename = self._get_model_filename(episode)
            episode_path = self.models_dir / episode_filename
            torch.save(checkpoint, episode_path)
            print(f"RL model saved to {episode_path}")
        
        # 最新モデル保存
        if save_latest:
            latest_filename = self._get_model_filename()
            latest_path = self.models_dir / latest_filename
            torch.save(checkpoint, latest_path)
            print(f"Latest RL model saved to {latest_path}")
        
        # 古いモデルファイルのクリーンアップ
        if self.config.get('cleanup_old_models', False):
            self.cleanup_old_models()
    
    def _save_model(self) -> None:
        """モデルの保存（後方互換性のため）"""
        # 学習終了時の保存として呼び出される
        self.save_model(self.episodes, 0.0, save_latest=True)
    
    def cleanup_old_models(self, keep_last_n: int = 5):
        """古いモデルファイルを削除"""
        if not self.models_dir.exists():
            return
        
        config_hash = self._get_config_hash()
        pattern = f"rl_model_{config_hash}_episode_"
        
        # エピソード別ファイルを取得してソート
        epoch_files = []
        for model_file in self.models_dir.glob("rl_model_*.pt"):
            filename = model_file.name
            if filename.startswith(pattern):
                try:
                    episode_str = filename[len(pattern):-3]  # .ptを除く
                    episode_num = int(episode_str)
                    epoch_files.append((episode_num, filename))
                except ValueError:
                    continue
        
        # エピソード番号でソート
        epoch_files.sort(key=lambda x: x[0])
        
        # 古いファイルを削除
        if len(epoch_files) > keep_last_n:
            for episode_num, filename in epoch_files[:-keep_last_n]:
                filepath = self.models_dir / filename
                if filepath.exists():
                    filepath.unlink()
                    print(f"Removed old RL model: {filename}")
    
    def load_model(self, model_path: str) -> None:
        """
        保存されたモデルの読み込み（後方互換性のため）
        
        Args:
            model_path: モデルファイルのパス
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 追加パラメータの復元
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        
        print(f"RL model loaded from {model_path}")
    
    def list_available_models(self):
        """利用可能なモデルを一覧表示（外部から呼び出し可能）"""
        self._show_available_models()
    
    def _get_actual_load_factor(self, data_idx: int) -> float:
        """
        GCNと同じ方法で実際のデータから負荷率を取得
        
        Args:
            data_idx: データインデックス
            
        Returns:
            実際の負荷率（gt_load_factor）
        """
        # データセットリーダーを使用してバッチを取得（GCNと同じ方法）
        batch_size = 1
        dataset = DatasetReader(self.config.get('num_test_data', 20), batch_size, 'test')
        
        # 指定されたdata_idxのデータを取得
        for i, batch in enumerate(dataset):
            if i == data_idx:
                # GCNと同じ方法: np.mean(batch.load_factor)
                gt_load_factor = np.mean(batch.load_factor)
                # print(f"    Actual data: data_idx={data_idx}, gt_load_factor={gt_load_factor:.6f}")
                return gt_load_factor
        
        # データが見つからない場合はエラー
        raise ValueError(f"Could not find actual load factor for data_idx={data_idx}. Expected data range: 0 to {self.config.get('num_test_data', 20)-1}")
    
    def _read_approximation_from_file(self, rl_max_load: float, data_idx: int) -> float:
        """
        既存の厳密解ファイルから近似率を計算（GCNと同じ方法）
        CSVファイル形式: 各行は `load_factor,time` (行番号=data_idx+1)
        GCNの計算方法: gt_load_factor / predicted_load_factor * 100
        """
        try:
            exact_solution_file = './data/test_data/exact_solution.csv'
            
            with open(exact_solution_file, 'r') as f:
                lines = f.readlines()
                
                # data_idxは0-indexed、CSVも0-indexedとして扱う
                if 0 <= data_idx < len(lines):
                    line = lines[data_idx].strip()
                    
                    # 形式: "0.26725906,0.10788202285766602" 
                    parts = line.split(',')
                    if len(parts) >= 1:
                        gt_load_factor = float(parts[0])
                        
                        # デバッグ情報
                        print(f"    CSV found: data_idx={data_idx}, gt_load={gt_load_factor:.6f}, rl_load={rl_max_load:.6f}")
                        
                        # GCNと同じ計算方法: 厳密解 / 予測値 * 100
                        if rl_max_load > 0:
                            approximation_rate = gt_load_factor / rl_max_load * 100
                            print(f"    Calculated approximation rate: {approximation_rate:.2f}%")
                            return approximation_rate
                        else:
                            return 0.0
                else:
                    print(f"    CSV debug: data_idx={data_idx} out of range (file has {len(lines)} lines)")
            
            # データが見つからない場合は0.0を返す（GCNと同じ処理）
            return 0.0
            
        except Exception as e:
            print(f"Warning: Error reading exact solution file: {e}")
            return 0.0