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

    def _compute_completion_and_approx(self, pred_paths, batch_commodities, gt_load_factors_per_sample, edge_capacity):
        """
        Comp%・CompSample%・Approx% を SeqFlowRL と同じ定義で計算する。

        - Comp%: コモディティ単位の到達率
        - CompSample%: 全コモディティ到達サンプルの割合
        - Approx%: 全到達サンプルのみ per-sample (gt_lf / model_lf) を平均

        Args:
            pred_paths: List[List[List[int]]] [batch][commodity] = node list
            batch_commodities: Tensor [B, C, 3]
            gt_load_factors_per_sample: array-like [B]
            edge_capacity: Tensor [B, V, V]
        """
        batch_size = len(pred_paths)
        num_commodities = batch_commodities.shape[1]

        total_commodities = 0
        complete_commodities = 0
        complete_mask = []
        model_load_factors = []

        for b in range(batch_size):
            all_complete = True
            for c in range(num_commodities):
                total_commodities += 1
                path = pred_paths[b][c] if b < len(pred_paths) and c < len(pred_paths[b]) else []
                dst = int(batch_commodities[b, c, 1].item())
                if len(path) > 0 and path[-1] == dst:
                    complete_commodities += 1
                else:
                    all_complete = False
            complete_mask.append(all_complete)

            # Per-sample load factor from pred_paths
            cap_b = edge_capacity[b]
            edge_usage = torch.zeros_like(cap_b)
            for c in range(num_commodities):
                path = pred_paths[b][c] if b < len(pred_paths) and c < len(pred_paths[b]) else []
                demand = float(batch_commodities[b, c, 2].item())
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if 0 <= u < cap_b.shape[0] and 0 <= v < cap_b.shape[1]:
                        edge_usage[u, v] += demand
            valid = cap_b > 0
            lf = (edge_usage[valid] / cap_b[valid]).max().item() if valid.any() else 0.0
            model_load_factors.append(lf)

        comp_rate = complete_commodities / total_commodities * 100 if total_commodities > 0 else 0.0
        comp_sample_rate = sum(complete_mask) / batch_size * 100 if batch_size > 0 else 0.0

        approx_ratios = []
        for b in range(batch_size):
            if complete_mask[b] and model_load_factors[b] > 1e-8:
                gt_lf = float(gt_load_factors_per_sample[b])
                if gt_lf > 0:
                    approx_ratios.append(gt_lf / model_load_factors[b] * 100)
        approx_rate = float(np.mean(approx_ratios)) if approx_ratios else 0.0

        return comp_rate, comp_sample_rate, approx_rate

    def evaluate(self, net, master_bar, mode='test'):
        """モデルを評価"""
        net.eval()
        num_data = getattr(self.config, f'num_{mode}_data')
        batch_size = self.config.batch_size
        dataset = DatasetReader(num_data, batch_size, mode, self.config)
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
        all_comp_rates = []
        all_comp_sample_rates = []
        all_approx_rates = []

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

                device = next(net.parameters()).device
                x_edges = x_edges.to(device)
                x_edges_capacity = x_edges_capacity.to(device)
                x_nodes = x_nodes.to(device)
                y_edges = y_edges.to(device)
                batch_commodities = batch_commodities.to(device)
                x_commodities = x_commodities.to(device)

                if self.strategy is not None:
                    batch_data = self.strategy.prepare_batch_data(batch, device)
                    loss, metrics = self.strategy.compute_loss(net, batch_data, device)
                    mean_maximum_load_factor = metrics.get('mean_load_factor', 0.0)
                    pred_paths = metrics.get('pred_paths', None)
                else:
                    if type(edge_cw) != torch.Tensor:
                        edge_labels = y_edges.cpu().numpy().flatten()
                        edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

                    y_preds, loss = net.forward(x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges, edge_cw)
                    loss = loss.mean()
                    _, _ = edge_error(y_preds, y_edges, x_edges)

                    beam_search = BeamsearchUELB(
                        y_preds, self.config.beam_size, batch_size, x_edges_capacity, batch_commodities, torch.float, torch.long, mode_strict=True)
                    pred_paths, is_feasible = beam_search.search()
                    mean_maximum_load_factor, _ = mean_feasible_load_factor(batch_size, self.config.num_commodities, self.config.num_nodes, pred_paths, x_edges_capacity, batch_commodities)

                gt_load_factor = np.mean(batch.load_factor)

                if mean_maximum_load_factor > 1 or mean_maximum_load_factor == 0:
                    infeasible_count += 1
                else:
                    feasible_count += 1

                # Comp% / CompSample% / Approx% (SeqFlowRL と同じ定義)
                if pred_paths is not None:
                    comp_rate, comp_sample_rate, approx_rate = self._compute_completion_and_approx(
                        pred_paths, batch_commodities, batch.load_factor, x_edges_capacity
                    )
                    all_comp_rates.append(comp_rate)
                    all_comp_sample_rates.append(comp_sample_rate)
                    if approx_rate > 0:
                        all_approx_rates.append(approx_rate)

                running_nb_data += batch_size
                running_loss += batch_size * loss.data.item() * self.config.accumulation_steps
                running_mean_maximum_load_factor += mean_maximum_load_factor
                running_gt_load_factor += gt_load_factor
                running_nb_batch += 1

            loss = running_loss / running_nb_data
            infeasible_rate = infeasible_count / (feasible_count + infeasible_count) * 100 if (feasible_count + infeasible_count) > 0 else 0

            if running_nb_batch != 0:
                mean_gt_load_factor = running_gt_load_factor / running_nb_batch
                epoch_mean_maximum_load_factor = running_mean_maximum_load_factor / running_nb_batch
            else:
                mean_gt_load_factor = 0
                epoch_mean_maximum_load_factor = 0

            mean_comp_rate = float(np.mean(all_comp_rates)) if all_comp_rates else 0.0
            mean_comp_sample_rate = float(np.mean(all_comp_sample_rates)) if all_comp_sample_rates else 0.0
            approximation_rate = float(np.mean(all_approx_rates)) if all_approx_rates else 0.0

        return (time.time() - start_test, loss, epoch_mean_maximum_load_factor,
                mean_gt_load_factor, approximation_rate, infeasible_rate,
                mean_comp_rate, mean_comp_sample_rate)

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
        val_time, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate, val_comp_rate, val_comp_sample_rate = self.evaluate(
            trainer.get_model(), None, mode='val'
        )
        metrics_logger.log_val_metrics(val_approximation_rate, val_time,
                                       comp_rate=val_comp_rate, comp_sample_rate=val_comp_sample_rate)

        print('v: ' + metrics_to_str(0, val_time, learning_rate, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate, val_comp_rate, val_comp_sample_rate))

        # テストの実行
        print("Running test...")
        test_time, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate, test_comp_rate, test_comp_sample_rate = self.evaluate(
            trainer.get_model(), None, mode='test'
        )
        metrics_logger.log_test_metrics(test_approximation_rate, test_time,
                                        comp_rate=test_comp_rate, comp_sample_rate=test_comp_sample_rate)

        print('T: ' + metrics_to_str(0, test_time, learning_rate, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate, test_comp_rate, test_comp_sample_rate))

        val_result = (val_time, val_loss, val_mean_maximum_load_factor, val_gt_load_factor, val_approximation_rate, val_infeasible_rate, val_comp_rate, val_comp_sample_rate)
        test_result = (test_time, test_loss, test_mean_maximum_load_factor, test_gt_load_factor, test_approximation_rate, test_infeasible_rate, test_comp_rate, test_comp_sample_rate)

        return val_result, test_result 