import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt

from config import get_config
from utils.graph_utils import *
from utils.exact_solution import SolveExactSolution
from utils.beamsearch_uelb import BeamsearchUELB
from utils.flow import Flow
from utils.data_maker import DataMaker
from utils.dataset_reader import DatasetReader
from utils.plot_utils import plot_uelb
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import create_edge_class_weights, mean_load_factor, mean_feasible_load_factor, edge_error, update_learning_rate
from sklearn.utils.class_weight import compute_class_weight
from utils.create_data_files import create_data_files

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter("ignore")
warnings.simplefilter('ignore', SparseEfficiencyWarning)

def set_gpu(config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    if torch.cuda.is_available():
        print(f"CUDA available, using GPU ID {config.gpu_id}")
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        torch.cuda.manual_seed(1)
    else:
        print("CUDA not available")
        dtypeFloat = torch.float
        dtypeLong = torch.long
        torch.manual_seed(1)
    return dtypeFloat, dtypeLong

def make_graph_dataset(config):
    for mode in ["val", "test", "train"]:
        create_data_files(config, data_mode=mode)
    # 厳密解の計算例（必要なら有効化）
    # for data in range(20):
    #     ...

def test_data_loading(config):
    mode = "test"
    num_data = getattr(config, f'num_{mode}_data')
    batch_size = config.batch_size
    dataset = DatasetReader(num_data, batch_size, mode)
    print(f"Number of batches of size {batch_size}: {dataset.max_iter}")
    batch = next(iter(dataset))
    print("edges shape:", batch.edges.shape)
    # ...他のprintは省略可...
    f = plt.figure(figsize=(5, 5))
    a = f.add_subplot(111)
    plot_uelb(a, batch.edges[0], batch.edges_target[0])
    plt.close(f)

def instantiate_model(config, dtypeFloat, dtypeLong):
    net = nn.DataParallel(ResidualGatedGCNModel(config, dtypeFloat, dtypeLong))
    if torch.cuda.is_available():
        net.cuda()
    print(net)
    nb_param = sum(np.prod(list(param.data.size())) for param in net.parameters())
    print('Number of parameters:', nb_param)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    print(optimizer)
    torch.autograd.set_detect_anomaly(True)
    return net, optimizer

def train_one_epoch(net, optimizer, config, master_bar):
    net.train()
    mode = "train"
    num_data = getattr(config, f'num_{mode}_data')
    batch_size = config.batch_size
    batches_per_epoch = config.batches_per_epoch
    accumulation_steps = config.accumulation_steps
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
        y_preds, loss = net.forward(x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges, edge_cw)
        loss = loss.mean() / accumulation_steps
        loss.backward()
        if (batch_num+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        err_edges = edge_error(y_preds, y_edges, x_edges)
        running_nb_data += batch_size
        running_loss += batch_size * loss.data.item() * accumulation_steps
        running_err_edges += batch_size * err_edges
    loss = running_loss / running_nb_data
    err_edges = running_err_edges / running_nb_data
    return time.time()-start_epoch, loss, err_edges

def test(net, config, master_bar, mode='test'):
    net.eval()
    num_data = getattr(config, f'num_{mode}_data')
    batch_size = config.batch_size
    batches_per_epoch = config.batches_per_epoch
    dataset = DatasetReader(num_data, batch_size, mode)
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
            if type(edge_cw) != torch.Tensor:
                edge_labels = y_edges.cpu().numpy().flatten()
                edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
            y_preds, loss = net.forward(x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges, edge_cw)
            loss = loss.mean()
            beam_search = BeamsearchUELB(
                y_preds, config.beam_size, batch_size, x_edges_capacity, batch_commodities, torch.float, torch.long, mode_strict=True)
            pred_paths, is_feasible = beam_search.search()
            mean_maximum_load_factor, _ = mean_feasible_load_factor(batch_size, config.num_commodities, config.num_nodes, pred_paths, x_edges_capacity, batch_commodities)
            if mean_maximum_load_factor > 1:
                mean_maximum_load_factor = 0
                gt_load_factor = 0
                infeasible_count += 1
            else:
                feasible_count += 1
                gt_load_factor = np.mean(batch.load_factor)
            running_nb_data += batch_size
            running_loss += batch_size * loss.data.item() * config.accumulation_steps
            running_mean_maximum_load_factor += mean_maximum_load_factor
            running_gt_load_factor += gt_load_factor
            running_nb_batch += 1
        loss = running_loss / running_nb_data
        infeasible_rate = infeasible_count / (feasible_count + infeasible_count) * 100 if (feasible_count + infeasible_count) > 0 else 0
        if feasible_count != 0:
            mean_gt_load_factor = running_gt_load_factor / feasible_count
            epoch_mean_maximum_load_factor = running_mean_maximum_load_factor / feasible_count
            approximation_rate = mean_gt_load_factor / epoch_mean_maximum_load_factor * 100 if epoch_mean_maximum_load_factor != 0 else 0
        else:
            mean_gt_load_factor = 0
            epoch_mean_maximum_load_factor = 0
            approximation_rate = 0
    return time.time()-start_test, loss, epoch_mean_maximum_load_factor, mean_gt_load_factor, approximation_rate, infeasible_rate

def ask_remake_dataset():
    ans = input("トレーニング・検証・テスト用データを再作成しますか？ (y/n): ").strip().lower()
    return ans in ["y", "yes"]

def main():
    config_path = "configs/default2.json"
    config = get_config(config_path)
    dtypeFloat, dtypeLong = set_gpu(config)
    if ask_remake_dataset():
        make_graph_dataset(config)
    # test_data_loading(config)  # 必要なら有効化
    net, optimizer = instantiate_model(config, dtypeFloat, dtypeLong)
    max_epochs = config.max_epochs
    val_every = config.val_every
    test_every = config.test_every
    learning_rate = config.learning_rate
    decay_rate = config.decay_rate
    val_loss_old = 1e6
    train_loss_list = []
    train_err_edges_list = []
    val_approximation_rate_list = []
    test_approximation_rate_list = []
    epoch_bar = master_bar(range(max_epochs))
    for epoch in epoch_bar:
        train_time, train_loss, train_err_edges = train_one_epoch(net, optimizer, config, epoch_bar)
        train_loss_list.append(train_loss)
        train_err_edges_list.append(train_err_edges)
        if epoch % val_every == 0 or epoch == max_epochs - 1:
            val_time, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate = test(net, config, epoch_bar, mode='val')
            val_approximation_rate_list.append(val_approximation_rate)
        if epoch % val_every == 0 and epoch != 0:
            learning_rate /= decay_rate
            optimizer = update_learning_rate(optimizer, learning_rate)
            val_loss_old = val_loss
        if epoch % test_every == 0 or epoch == max_epochs - 1:
            test_time, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate = test(net, config, epoch_bar, mode='test')
            test_approximation_rate_list.append(test_approximation_rate)
    print("Training finished.")

if __name__ == "__main__":
    main()
