import time
import torch
import numpy as np
from fastprogress import master_bar, progress_bar
from sklearn.utils.class_weight import compute_class_weight

from ..data_management.dataset_reader import DatasetReader
from ..algorithms.beamsearch_uelb import BeamsearchUELB
from ..models.model_utils import edge_error, mean_feasible_load_factor

class Evaluator:
    """評価を担当するクラス"""
    
    def __init__(self, config, dtypeFloat, dtypeLong):
        self.config = config
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
    
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
                
                if mean_maximum_load_factor > 1:
                    mean_maximum_load_factor = 0
                    gt_load_factor = 0
                    infeasible_count += 1
                else:
                    feasible_count += 1
                    gt_load_factor = np.mean(batch.load_factor)
                
                running_nb_data += batch_size
                running_loss += batch_size * loss.data.item() * self.config.accumulation_steps
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