import time
import torch
import torch.nn as nn
import numpy as np
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
        self.net, self.optimizer = self._instantiate_model()
    
    def _instantiate_model(self):
        """モデルとオプティマイザーを初期化"""
        net = nn.DataParallel(ResidualGatedGCNModel(self.config, self.dtypeFloat, self.dtypeLong))
        if torch.cuda.is_available():
            net.cuda()
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
        num_data = getattr(self.config, f'num_{mode}_data')
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