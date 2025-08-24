import time
import torch
import torch.nn as nn
import numpy as np
import os
import hashlib
import json
from fastprogress import master_bar, progress_bar
from sklearn.utils.class_weight import compute_class_weight

from ..models.gcn_model import ResidualGatedGCNModel
from ..data_management.dataset_reader import DatasetReader
from ..models.model_utils import edge_error, update_learning_rate

class Trainer:
    """トレーニングを担当するクラス"""
    
    def __init__(self, config, dtypeFloat, dtypeLong):
        self.config = config
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        self.models_dir = config.get('models_dir', './saved_models')
        self.net, self.optimizer = self._instantiate_model()
    
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
            
        print(net)
        nb_param = sum(np.prod(list(param.data.size())) for param in net.parameters())
        print('Number of parameters:', nb_param)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.config.learning_rate)
        print(optimizer)
        torch.autograd.set_detect_anomaly(True)
        return net, optimizer
    
    def train_one_epoch(self, master_bar):
        """1エポックのトレーニングを実行"""
        self.net.train()
        mode = "train"
        num_data = self.config.get(f'num_{mode}_data')
        batch_size = self.config.batch_size
        batches_per_epoch = self.config.batches_per_epoch
        accumulation_steps = self.config.accumulation_steps
        dataset = DatasetReader(num_data, batch_size, mode)
        
        if batches_per_epoch != -1:
            batches_per_epoch = min(batches_per_epoch, dataset.max_iter)
        else:
            batches_per_epoch = dataset.max_iter
        
        dataset = iter(dataset)
        edge_cw = None
        running_loss = 0.0
        running_err_edges = 0.0
        running_nb_data = 0
        start_epoch = time.time()
        
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
            device = next(self.net.parameters()).device
            x_edges = x_edges.to(device)
            x_edges_capacity = x_edges_capacity.to(device)
            x_nodes = x_nodes.to(device)
            y_edges = y_edges.to(device)
            batch_commodities = batch_commodities.to(device)
            x_commodities = x_commodities.to(device)
            
            if type(edge_cw) != torch.Tensor:
                edge_labels = y_edges.cpu().numpy().flatten()
                edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
            
            y_preds, loss = self.net.forward(x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges, edge_cw)
            loss = loss.mean() / accumulation_steps
            loss.backward()
            
            if (batch_num+1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            err_edges, _ = edge_error(y_preds, y_edges, x_edges)
            running_nb_data += batch_size
            running_loss += batch_size * loss.data.item() * accumulation_steps
            running_err_edges += batch_size * err_edges
        
        loss = running_loss / running_nb_data
        err_edges = running_err_edges / running_nb_data
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
        config_for_hash = {
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'mlp_layers': self.config.mlp_layers,
            'node_dim': self.config.node_dim,
            'voc_nodes_in': self.config.voc_nodes_in,
            'voc_nodes_out': self.config.voc_nodes_out,
            'voc_edges_in': self.config.voc_edges_in,
            'voc_edges_out': self.config.voc_edges_out,
            'aggregation': self.config.aggregation,
            'dropout_rate': self.config.get('dropout_rate', 0.0),
        }
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
                'voc_nodes_in': self.config.voc_nodes_in,
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