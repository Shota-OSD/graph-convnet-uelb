import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple

from src.gnn_ils.models.gnn_ils_model import GNNILSModel
from src.gnn_ils.environment.ils_environment import ILSEnvironment


class ILSA2CStrategy:
    """
    2段階 A2C Training Strategy for GNN-ILS。

    ILS改善ループ内で Level1 (コモディティ選択) + Level2 (パス選択) の
    ログ確率・Value・エントロピーを蓄積し、エピソード終了時に A2C 損失を計算する。

    損失:
        L = L_actor_l1 + L_actor_l2
            + value_loss_weight * L_critic
            - entropy_weight_l1 * H_l1
            - entropy_weight_l2 * H_l2
    """

    def __init__(self, model: GNNILSModel, config: dict):
        """
        Args:
            config:
                - learning_rate, weight_decay, adam_beta1, adam_beta2
                - entropy_weight_l1, entropy_weight_l2
                - value_loss_weight, gamma
                - normalize_advantages, grad_clip_norm
                - reward_mode, max_iterations, no_improve_patience
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        self.entropy_weight_l1 = config.get('entropy_weight_l1', 0.02)
        self.entropy_weight_l2 = config.get('entropy_weight_l2', 0.01)
        self.value_loss_weight = config.get('value_loss_weight', 0.5)
        self.gamma = config.get('gamma', 0.99)
        self.normalize_advantages = config.get('normalize_advantages', True)
        self.grad_clip_norm = config.get('grad_clip_norm', 1.0)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.0005),
            weight_decay=config.get('weight_decay', 0.0001),
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
        )

        if config.get('lr_scheduler', None) == 'step':
            decay_rate = config.get('decay_rate', 1.2)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=1.0 / decay_rate
            )
        else:
            self.scheduler = None

        self.env = ILSEnvironment(config)

    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        1サンプルの ILS エピソードを実行し、損失を計算・更新する。

        Args:
            batch_data:
                - x_nodes [1, V, C], x_commodities [1, C, 3],
                  x_edges_capacity [1, V, V], load_factor [1]
                - graph: nx.Graph, commodity_list: List[List[int]]

        Returns:
            metrics 辞書
        """
        self.model.train()

        trajectory = self._run_ils_episode(batch_data, deterministic=False)

        if len(trajectory['rewards']) == 0:
            return self._empty_metrics()

        loss, loss_components = self._compute_a2c_loss(trajectory)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        improvement = 0.0
        init_lf = trajectory['initial_load_factor']
        final_lf = trajectory['final_load_factor']
        if init_lf > 1e-8:
            improvement = (init_lf - final_lf) / init_lf * 100.0

        metrics = {
            **loss_components,
            'mean_reward': float(np.mean(trajectory['rewards'])),
            'final_load_factor': final_lf,
            'initial_load_factor': init_lf,
            'improvement': improvement,
            'num_iterations': len(trajectory['rewards']),
            'best_iteration': trajectory['best_iteration'],
        }
        return metrics

    def _run_ils_episode(
        self, batch_data: Dict[str, Any], deterministic: bool
    ) -> Dict:
        """
        ILS エピソードを実行し、trajectory を収集する。

        Returns:
            trajectory 辞書 (log_probs, values, rewards など)
        """
        state = self.env.reset(
            G=batch_data['graph'],
            commodity_list=batch_data['commodity_list'],
            x_nodes=batch_data['x_nodes'],
            x_commodities=batch_data['x_commodities'],
            x_edges_capacity=batch_data['x_edges_capacity'],
        )

        trajectory: Dict[str, List] = {
            'log_probs_l1': [],
            'log_probs_l2': [],
            'entropies_l1': [],
            'entropies_l2': [],
            'state_values': [],
            'rewards': [],
            'initial_load_factor': state['load_factor'],
            'final_load_factor': state['load_factor'],
        }

        done = False
        while not done:
            # デバイスへ転送
            x_nodes = state['x_nodes'].to(self.device)
            x_commodities = state['x_commodities'].to(self.device)
            x_edges_capacity = state['x_edges_capacity'].to(self.device)
            x_edges_usage = state['x_edges_usage'].to(self.device)
            commodity_mask = state['commodity_mask'].to(self.device)

            # 全コモディティが交換不可なら終了
            if not commodity_mask.any():
                break

            # GNN エンコード
            node_features, edge_features, graph_embedding = self.model.encode(
                x_nodes, x_commodities, x_edges_capacity, x_edges_usage
            )

            # Critic: 状態価値
            state_value = self.model.get_value(node_features, graph_embedding)

            # Level1: コモディティ選択
            demands = state['x_commodities'][:, :, 2].to(self.device)  # [1, C]
            current_assignment_batch = [state['current_assignment']]     # [1][C][path_length]
            selected_commodity, log_prob_l1, entropy_l1 = self.model.select_commodity(
                edge_features, current_assignment_batch, demands,
                commodity_mask, deterministic=deterministic
            )
            c_idx = selected_commodity[0].item()

            # Level2: パス選択
            path_mask = self.env.get_path_mask(c_idx).to(self.device)
            candidate_paths = [state['path_pool'][c_idx]]  # [1][P_c][path_length]

            current_paths_batch = [state['current_assignment'][c_idx]]  # [1][path_length]
            demand_c = demands[:, c_idx]  # [1] (生の demand、正規化はモデル内で行う)
            selected_path_idx, log_prob_l2, entropy_l2 = self.model.select_path(
                edge_features, selected_commodity,
                candidate_paths, current_paths_batch, demand_c,
                path_mask, deterministic=deterministic
            )

            # 環境を1ステップ進める
            new_state, reward, done, info = self.env.step(
                c_idx, selected_path_idx[0].item()
            )

            trajectory['log_probs_l1'].append(log_prob_l1[0])
            trajectory['log_probs_l2'].append(log_prob_l2[0])
            trajectory['entropies_l1'].append(entropy_l1[0])
            trajectory['entropies_l2'].append(entropy_l2[0])
            trajectory['state_values'].append(state_value[0])
            trajectory['rewards'].append(reward)

            state = new_state

        trajectory['final_load_factor'] = state['load_factor']

        # Best-so-far 情報を追加
        best_solution = self.env.get_best_solution()
        trajectory['best_load_factor'] = best_solution['best_load_factor']
        trajectory['best_iteration'] = best_solution['best_iteration']

        return trajectory

    def _compute_a2c_loss(
        self, trajectory: Dict
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        2段階 A2C 損失を計算する。

        1. discounted returns R_t = sum_{k} gamma^k * r_{t+k}
        2. advantage A_t = R_t - V(s_t).detach()
        3. L_actor_l1 = -mean(log_prob_l1 * A_t)
        4. L_actor_l2 = -mean(log_prob_l2 * A_t)
        5. L_critic   = MSE(V(s_t), R_t)
        6. entropy bonus
        """
        rewards = trajectory['rewards']
        T = len(rewards)

        # Discounted returns
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        log_probs_l1 = torch.stack(trajectory['log_probs_l1'])  # [T]
        log_probs_l2 = torch.stack(trajectory['log_probs_l2'])  # [T]
        entropies_l1 = torch.stack(trajectory['entropies_l1'])  # [T]
        entropies_l2 = torch.stack(trajectory['entropies_l2'])  # [T]
        state_values = torch.stack(trajectory['state_values'])  # [T]

        # Advantage
        advantages = returns_t - state_values.detach()
        if self.normalize_advantages and T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor losses
        actor_l1_loss = -(log_probs_l1 * advantages).mean()
        actor_l2_loss = -(log_probs_l2 * advantages).mean()

        # Critic loss
        critic_loss = F.mse_loss(state_values, returns_t)

        # Entropy
        entropy_l1_mean = entropies_l1.mean()
        entropy_l2_mean = entropies_l2.mean()

        total_loss = (
            actor_l1_loss
            + actor_l2_loss
            + self.value_loss_weight * critic_loss
            - self.entropy_weight_l1 * entropy_l1_mean
            - self.entropy_weight_l2 * entropy_l2_mean
        )

        loss_components = {
            'total_loss': total_loss.item(),
            'actor_l1_loss': actor_l1_loss.item(),
            'actor_l2_loss': actor_l2_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_l1': entropy_l1_mean.item(),
            'entropy_l2': entropy_l2_mean.item(),
        }
        return total_loss, loss_components

    def eval_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        評価ステップ (勾配なし、deterministic)。

        Returns:
            metrics 辞書
        """
        self.model.eval()

        with torch.no_grad():
            trajectory = self._run_ils_episode(batch_data, deterministic=True)

        init_lf = trajectory['initial_load_factor']
        best_lf = trajectory['best_load_factor']
        best_iter = trajectory['best_iteration']
        improvement = 0.0
        if init_lf > 1e-8:
            improvement = (init_lf - best_lf) / init_lf * 100.0

        # Approximation ratio (グラウンドトゥルースと比較、best-so-far を使用)
        approximation_ratio = None
        gt_lf = batch_data.get('load_factor_scalar')
        if gt_lf is not None and gt_lf > 1e-8 and best_lf > 1e-8:
            approximation_ratio = (gt_lf / best_lf) * 100.0

        return {
            'final_load_factor': best_lf,
            'initial_load_factor': init_lf,
            'improvement': improvement,
            'num_iterations': len(trajectory['rewards']),
            'best_iteration': best_iter,
            'complete_rate': 100.0,   # パスプール構造的保証
            'approximation_ratio': approximation_ratio,
        }

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            'total_loss': 0.0,
            'actor_l1_loss': 0.0,
            'actor_l2_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_l1': 0.0,
            'entropy_l2': 0.0,
            'mean_reward': 0.0,
            'final_load_factor': 0.0,
            'initial_load_factor': 0.0,
            'improvement': 0.0,
            'num_iterations': 0,
        }
