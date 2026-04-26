import os
import time
import json
import torch
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.gnn_ils.models.gnn_ils_model import GNNILSModel
from src.gnn_ils.training.ils_a2c_strategy import ILSA2CStrategy
from src.common.types import validate_batch_types
from src.common.config.paths import get_model_root


class GNNILSTrainer:
    """
    GNN-ILS Trainer。

    SeqFlowRLTrainer と同様の構成だが、以下が異なる:
    - batch_size=1 でサンプル単位に ILS エピソードを実行
    - samples_per_epoch で1エポックあたりのサンプル数を制限
    - ILS 改善ループ統計を追加で記録
    """

    def __init__(self, config: dict, dtypeFloat=torch.float32, dtypeLong=torch.long):
        self.config = config
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        self.device = self._setup_device()
        self.model = self._instantiate_model()
        self.strategy = ILSA2CStrategy(self.model, config)

        print(f"✓ Using 2-Level A2C Training Strategy (GNN-ILS)")
        print(f"  - entropy_weight_l1: {config.get('entropy_weight_l1', 0.02)}")
        print(f"  - entropy_weight_l2: {config.get('entropy_weight_l2', 0.01)}")
        print(f"  - value_loss_weight: {config.get('value_loss_weight', 0.5)}")
        print(f"  - max_iterations:    {config.get('max_iterations', 50)}")

        checkpoint_dir_cfg = config.get('checkpoint_dir')
        if checkpoint_dir_cfg:
            self.checkpoint_dir = Path(checkpoint_dir_cfg).expanduser()
        else:
            self.checkpoint_dir = get_model_root(config) / 'gnn_ils'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_every = config.get('save_every', 10)
        self.save_best_only = config.get('save_best_only', True)

        self.log_dir = Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'gnn_ils_training_{timestamp}.txt'

        self.current_epoch = 0
        self.best_val_load_factor = float('inf')
        self.training_history: Dict = {
            'train_loss': [],
            'train_reward': [],
            'train_load_factor': [],
            'train_improvement': [],
            'train_num_iterations': [],
            'train_approx_ratio': [],
            'train_best_iteration': [],
            'val_load_factor': [],
            'val_improvement': [],
            'val_num_iterations': [],
            'val_approx_ratio': [],
            'val_best_iteration': [],
            'learning_rate': [],
            'epoch_times': [],
        }
        self.training_start_time: Optional[float] = None
        self._val_loader = None

    def _setup_device(self) -> torch.device:
        use_gpu = self.config.get('use_gpu', True)
        gpu_id = self.config.get('gpu_id', '0')
        if use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            print(f"✓ Using GPU: {gpu_id}")
        else:
            device = torch.device('cpu')
            print(f"✓ Using CPU")
        return device

    def _instantiate_model(self) -> GNNILSModel:
        model = GNNILSModel(self.config, self.dtypeFloat, self.dtypeLong)
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*70}")
        print(f"MODEL SUMMARY (GNN-ILS)")
        print(f"{'='*70}")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Hidden dim:           {self.config.get('hidden_dim', 128)}")
        print(f"  GCN layers:           {self.config.get('num_layers', 8)}")
        print(f"  Max candidate paths:  {self.config.get('max_candidate_paths', 15)}")
        print(f"{'='*70}\n")

        return model

    def train(self, train_loader, val_loader=None, num_epochs=None) -> Dict:
        """
        メイン学習ループ。

        Args:
            train_loader: DatasetReader (batch_size=1)
            val_loader:   DatasetReader (batch_size=1, optional)
            num_epochs:   エポック数 (config['max_epochs'] で上書き可)

        Returns:
            training_history 辞書
        """
        if num_epochs is None:
            num_epochs = self.config.get('max_epochs', 50)
        self._val_loader = val_loader

        print(f"\n{'='*70}")
        print(f"STARTING TRAINING (GNN-ILS)")
        print(f"{'='*70}")
        print(f"  Epochs:           {num_epochs}")
        print(f"  Samples/epoch:    {self.config.get('samples_per_epoch', train_loader.max_iter)}")
        print(f"  Val loader:       {'Yes' if val_loader else 'No'}")
        print(f"  Log file:         {self.log_file}")
        print(f"{'='*70}\n")

        self.training_start_time = time.time()
        val_every = self.config.get('val_every', 5)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            train_metrics = self._train_epoch(train_loader, epoch)

            val_metrics = None
            if val_loader and (epoch + 1) % val_every == 0:
                val_metrics = self._validate_epoch(val_loader, epoch)

            self.strategy.step_scheduler()
            current_lr = self.strategy.get_current_lr()

            epoch_time = time.time() - epoch_start
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr, epoch_time)
            self._update_history(train_metrics, val_metrics, current_lr, epoch_time)

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)

            if val_metrics is not None and self.save_best_only:
                if val_metrics['mean_load_factor'] < self.best_val_load_factor:
                    self.best_val_load_factor = val_metrics['mean_load_factor']
                    self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
                    print(f"  ★ New best val load factor: {self.best_val_load_factor:.4f}")

        total_time = time.time() - self.training_start_time
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED (GNN-ILS)")
        print(f"{'='*70}")
        print(f"  Best val load factor: {self.best_val_load_factor:.4f}")
        print(f"  Total time:           {total_time:.2f}s")
        print(f"{'='*70}\n")

        self._save_training_log(total_time)
        return self.training_history

    def _prepare_batch(self, batch: Any) -> Dict:
        """
        DotDict バッチを ILS 用辞書に変換する。

        DatasetReader が返す行列から NetworkX グラフと commodity_list を復元する。
        """
        # 行列 → テンソル変換 (batch_size=1 前提)
        batch_data = {
            'x_nodes': torch.from_numpy(batch.nodes).long(),
            'x_commodities': torch.from_numpy(batch.commodities).float(),
            'x_edges_capacity': torch.from_numpy(batch.edges_capacity).float(),
            'x_edges': torch.from_numpy(batch.edges).long(),
            'load_factor': torch.from_numpy(batch.load_factor).float(),
        }

        # グラフ復元 (インデックス0のサンプルを使用)
        adj_np = batch.edges[0].astype(int)         # [V, V]
        cap_np = batch.edges_capacity[0]             # [V, V]
        batch_data['graph'] = self._reconstruct_graph(adj_np, cap_np)

        # コモディティリスト復元
        comm_np = batch.commodities[0].astype(int)  # [C, 3]
        batch_data['commodity_list'] = comm_np.tolist()

        # スカラー gt 負荷率 (eval 用)
        batch_data['load_factor_scalar'] = float(batch.load_factor[0])

        return batch_data

    def _reconstruct_graph(self, adj_np: np.ndarray, cap_np: np.ndarray) -> nx.DiGraph:
        """隣接行列と容量行列から NetworkX 有向グラフを復元する。"""
        V = adj_np.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(V))
        for u in range(V):
            for v in range(V):
                if adj_np[u, v] > 0:
                    G.add_edge(u, v, weight=1.0, capacity=float(cap_np[u, v]))
        return G

    def _train_epoch(self, train_loader, epoch: int) -> Dict:
        """1エポックの学習。"""
        self.model.train()

        samples_per_epoch = self.config.get('samples_per_epoch', None)
        num_batches = train_loader.max_iter
        if samples_per_epoch is not None:
            num_batches = min(num_batches, samples_per_epoch)

        log_every = self.config.get('log_every', 10)

        epoch_metrics: Dict = {
            'total_loss': [],
            'actor_l1_loss': [],
            'actor_l2_loss': [],
            'critic_loss': [],
            'mean_reward': [],
            'final_load_factor': [],
            'improvement': [],
            'num_iterations': [],
            'approximation_ratio': [],
            'best_iteration': [],
        }

        dataset_iter = iter(train_loader)

        for batch_idx in range(num_batches):
            try:
                batch = next(dataset_iter)
            except StopIteration:
                break

            batch_data = self._prepare_batch(batch)
            batch_data = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_data.items()
            }

            metrics = self.strategy.train_step(batch_data)

            for key in epoch_metrics:
                if key in metrics and metrics[key] is not None:
                    epoch_metrics[key].append(metrics[key])

            if (batch_idx + 1) % log_every == 0:
                print(f"  [{epoch + 1}][{batch_idx + 1}/{num_batches}] "
                      f"Loss: {metrics.get('total_loss', 0):.4f} | "
                      f"LF: {metrics.get('final_load_factor', 0):.4f} | "
                      f"Improvement: {metrics.get('improvement', 0):.1f}% | "
                      f"Iters: {metrics.get('num_iterations', 0)}")

        avg = {}
        for k, v in epoch_metrics.items():
            if k == 'approximation_ratio':
                valid = [x for x in v if x is not None]
                avg[k] = np.mean(valid) if valid else None
            elif not v:
                avg[k] = 0.0
            else:
                avg[k] = float(np.mean(v))

        # 統一キー名
        avg['mean_load_factor'] = avg.get('final_load_factor', 0.0)
        return avg

    def _validate_epoch(self, val_loader, epoch: int) -> Optional[Dict]:
        """1エポックの検証。"""
        self.model.eval()

        num_batches = val_loader.max_iter

        if num_batches == 0:
            return None

        epoch_metrics: Dict = {
            'final_load_factor': [],
            'improvement': [],
            'num_iterations': [],
            'approximation_ratio': [],
            'best_iteration': [],
        }

        dataset_iter = iter(val_loader)

        for batch_idx in range(num_batches):
            try:
                batch = next(dataset_iter)
            except StopIteration:
                break

            batch_data = self._prepare_batch(batch)
            batch_data = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_data.items()
            }

            metrics = self.strategy.eval_step(batch_data)

            for key in epoch_metrics:
                if key in metrics and metrics[key] is not None:
                    epoch_metrics[key].append(metrics[key])

        if not epoch_metrics['final_load_factor']:
            return None

        avg = {}
        for k, v in epoch_metrics.items():
            if k == 'approximation_ratio':
                valid = [x for x in v if x is not None]
                avg[k] = np.mean(valid) if valid else None
            elif not v:
                avg[k] = 0.0
            else:
                avg[k] = float(np.mean(v))

        avg['mean_load_factor'] = avg.get('final_load_factor', 0.0)
        avg['complete_rate'] = 100.0
        avg['complete_sample_rate'] = 100.0
        return avg

    def _log_epoch(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        print(f"\nEpoch {epoch + 1}/{self.config.get('max_epochs', 50)} "
              f"| Time: {epoch_time:.2f}s | LR: {lr:.6f}")

        approx_str = ""
        if train_metrics.get('approximation_ratio') is not None:
            approx_str = f" | Approx: {train_metrics['approximation_ratio']:.2f}%"

        print(f"  Train - Loss: {train_metrics.get('total_loss', 0):.4f} | "
              f"LF: {train_metrics.get('mean_load_factor', 0):.4f} | "
              f"Improve: {train_metrics.get('improvement', 0):.1f}% | "
              f"Iters: {train_metrics.get('num_iterations', 0):.1f} | "
              f"BestIter: {train_metrics.get('best_iteration', 0):.1f}"
              f"{approx_str}")

        if val_metrics is not None:
            val_approx_str = ""
            if val_metrics.get('approximation_ratio') is not None:
                val_approx_str = f" | Approx: {val_metrics['approximation_ratio']:.2f}%"
            print(f"  Val   - LF: {val_metrics.get('mean_load_factor', 0):.4f} | "
                  f"Improve: {val_metrics.get('improvement', 0):.1f}% | "
                  f"Iters: {val_metrics.get('num_iterations', 0):.1f} | "
                  f"BestIter: {val_metrics.get('best_iteration', 0):.1f}"
                  f"{val_approx_str}")

    def _update_history(self, train_metrics, val_metrics, lr, epoch_time):
        self.training_history['train_loss'].append(train_metrics.get('total_loss', 0.0))
        self.training_history['train_reward'].append(train_metrics.get('mean_reward', 0.0))
        self.training_history['train_load_factor'].append(train_metrics.get('mean_load_factor', 0.0))
        self.training_history['train_improvement'].append(train_metrics.get('improvement', 0.0))
        self.training_history['train_num_iterations'].append(train_metrics.get('num_iterations', 0.0))
        self.training_history['train_approx_ratio'].append(train_metrics.get('approximation_ratio'))
        self.training_history['train_best_iteration'].append(train_metrics.get('best_iteration', 0.0))
        self.training_history['learning_rate'].append(lr)
        self.training_history['epoch_times'].append(epoch_time)

        if val_metrics is not None:
            self.training_history['val_load_factor'].append(val_metrics.get('mean_load_factor', 0.0))
            self.training_history['val_improvement'].append(val_metrics.get('improvement', 0.0))
            self.training_history['val_num_iterations'].append(val_metrics.get('num_iterations', 0.0))
            self.training_history['val_approx_ratio'].append(val_metrics.get('approximation_ratio'))
            self.training_history['val_best_iteration'].append(val_metrics.get('best_iteration', 0.0))

    def _save_checkpoint(self, epoch, train_metrics, val_metrics, is_best=False):
        config_dict = dict(self.config) if hasattr(self.config, 'items') else self.config

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.strategy.optimizer.state_dict(),
            'config': config_dict,
            'training_history': self.training_history,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            print(f"  → Saving best model to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            print(f"  → Saving checkpoint to {path}")

        torch.save(checkpoint, path)
        torch.save(checkpoint, self.checkpoint_dir / 'latest_model.pt')

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.strategy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', self.training_history)
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch + 1}")
        return checkpoint

    def _save_training_log(self, total_training_time: float):
        with open(self.log_file, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("GNN-ILS TRAINING RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"Config: {self.config.get('expt_name', 'gnn_ils_base')}\n\n")

            # Final metrics
            f.write("FINAL METRICS:\n")
            if self.training_history['train_load_factor']:
                f.write(f"  Final Train LF:       {self.training_history['train_load_factor'][-1]:.4f}\n")
                f.write(f"  Final Train Improve:  {self.training_history['train_improvement'][-1]:.2f}%\n")
            if self.training_history['val_load_factor']:
                f.write(f"  Best Val LF:          {self.best_val_load_factor:.4f}\n")
            valid_val_approx = [x for x in self.training_history['val_approx_ratio'] if x is not None]
            if valid_val_approx:
                f.write(f"  Best Val Approx:      {np.max(valid_val_approx):.2f}%\n")
            f.write("\n")

            # Time metrics
            f.write("TIME METRICS:\n")
            f.write(f"  Total Training Time:  {total_training_time:.2f}s\n")
            if self.training_history['epoch_times']:
                f.write(f"  Avg Time per Epoch:   {np.mean(self.training_history['epoch_times']):.2f}s\n")
            f.write(f"  Total Epochs:         {len(self.training_history['train_loss'])}\n")
            f.write("\n")

            # Configuration
            f.write("CONFIGURATION:\n")
            f.write(f"  max_iterations:       {self.config.get('max_iterations', 50)}\n")
            f.write(f"  no_improve_patience:  {self.config.get('no_improve_patience', 10)}\n")
            f.write(f"  K:                    {self.config.get('K', 10)}\n")
            f.write(f"  max_candidate_paths:  {self.config.get('max_candidate_paths', 15)}\n")
            f.write(f"  samples_per_epoch:    {self.config.get('samples_per_epoch', 100)}\n")
            f.write(f"  learning_rate:        {self.config.get('learning_rate', 0.0005)}\n")
            f.write(f"  hidden_dim:           {self.config.get('hidden_dim', 128)}\n")
            f.write(f"  num_layers:           {self.config.get('num_layers', 8)}\n")
            f.write(f"  entropy_weight_l1:    {self.config.get('entropy_weight_l1', 0.02)}\n")
            f.write(f"  entropy_weight_l2:    {self.config.get('entropy_weight_l2', 0.01)}\n")
            f.write(f"  value_loss_weight:    {self.config.get('value_loss_weight', 0.5)}\n")
            f.write(f"  gamma:                {self.config.get('gamma', 0.99)}\n")
            f.write(f"  grad_clip_norm:       {self.config.get('grad_clip_norm', 1.0)}\n")
            f.write("\n")

            # Detailed train epoch results table
            f.write("=" * 50 + "\n")
            f.write("DETAILED EPOCH RESULTS (TRAIN)\n")
            f.write("=" * 50 + "\n")
            header = (f"{'Epoch':<8}{'Loss':<12}{'Reward':<12}{'LF':<10}"
                      f"{'Improve%':<12}{'Iters':<8}{'Approx%':<12}{'Time(s)':<10}\n")
            f.write(header)
            f.write("-" * 84 + "\n")

            for i in range(len(self.training_history['train_loss'])):
                loss    = self.training_history['train_loss'][i]
                reward  = self.training_history['train_reward'][i]
                lf      = self.training_history['train_load_factor'][i]
                improve = self.training_history['train_improvement'][i]
                iters   = self.training_history['train_num_iterations'][i]
                approx  = self.training_history['train_approx_ratio'][i]
                t       = self.training_history['epoch_times'][i]
                approx_str = f"{approx:.2f}" if approx is not None else "N/A"
                f.write(f"{i+1:<8}{loss:<12.4f}{reward:<12.4f}{lf:<10.4f}"
                        f"{improve:<12.1f}{iters:<8.1f}{approx_str:<12}{t:<10.2f}\n")

            f.write("-" * 84 + "\n")
            f.write("\n")

            # Detailed val epoch results table
            if self.training_history['val_load_factor']:
                val_every = self.config.get('val_every', 5)
                f.write("=" * 50 + "\n")
                f.write("DETAILED EPOCH RESULTS (VAL)\n")
                f.write("=" * 50 + "\n")
                f.write(f"{'Epoch':<8}{'Val LF':<12}{'Improve%':<12}{'Iters':<8}{'Approx%':<12}\n")
                f.write("-" * 52 + "\n")

                for i, (lf, improve, iters, approx) in enumerate(zip(
                    self.training_history['val_load_factor'],
                    self.training_history['val_improvement'],
                    self.training_history['val_num_iterations'],
                    self.training_history['val_approx_ratio'],
                )):
                    epoch_num = (i + 1) * val_every
                    approx_str = f"{approx:.2f}" if approx is not None else "N/A"
                    f.write(f"{epoch_num:<8}{lf:<12.4f}{improve:<12.1f}{iters:<8.1f}{approx_str:<12}\n")

                f.write("-" * 52 + "\n")
                f.write("\n")

            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            lf_hist = self.training_history['train_load_factor']
            if lf_hist:
                f.write(f"  Train Mean LF:        {np.mean(lf_hist):.4f}\n")
                f.write(f"  Train Min LF:         {np.min(lf_hist):.4f}\n")
                f.write(f"  Train Max LF:         {np.max(lf_hist):.4f}\n")
            iters_hist = self.training_history['train_num_iterations']
            if iters_hist:
                f.write(f"  Train Mean Iters:     {np.mean(iters_hist):.1f}\n")
            if self.training_history['val_load_factor']:
                f.write(f"  Best Val LF:          {self.best_val_load_factor:.4f}\n")
            if valid_val_approx:
                f.write(f"  Best Val Approx:      {np.max(valid_val_approx):.2f}%\n")
                f.write(f"  Mean Val Approx:      {np.mean(valid_val_approx):.2f}%\n")
            valid_train_approx = [x for x in self.training_history['train_approx_ratio'] if x is not None]
            if valid_train_approx:
                f.write(f"  Train Mean Approx:    {np.mean(valid_train_approx):.2f}%\n")
                f.write(f"  Train Best Approx:    {np.max(valid_train_approx):.2f}%\n")

        print(f"✓ Training log saved to: {self.log_file}")
