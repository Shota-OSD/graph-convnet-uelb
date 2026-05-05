import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple

from src.gnn_ils.models.gnn_ils_model import GNNILSModel
from src.gnn_ils.environment.ils_environment import ILSEnvironment


class ILSA2CStrategy:
    """
    2段階 A2C / PPO Training Strategy for GNN-ILS。

    ILS改善ループ内で Level1 (コモディティ選択) + Level2 (パス選択) の
    ログ確率・Value・エントロピーを蓄積し、エピソード終了時に損失を計算する。

    損失 (A2C):
        L = L_actor_l1 + L_actor_l2
            + value_loss_weight * L_critic
            - entropy_weight_l1 * H_l1
            - entropy_weight_l2 * H_l2

    PPO モード (use_ppo=True):
        クリップされた surrogate objective で複数回更新。
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
                - warmup_epochs, use_ppo, ppo_clip_eps, ppo_update_epochs
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

        # Critic warm-up
        self.warmup_epochs = config.get('warmup_epochs', 0)
        self.current_epoch = 0

        # PPO
        self.use_ppo = config.get('use_ppo', False)
        self.ppo_clip_eps = config.get('ppo_clip_eps', 0.2)
        self.ppo_update_epochs = config.get('ppo_update_epochs', 4)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.0005),
            weight_decay=config.get('weight_decay', 0.0001),
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
        )

        scheduler_type = config.get('lr_scheduler', None)
        if scheduler_type == 'step':
            decay_rate = config.get('decay_rate', 1.2)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=1.0 / decay_rate
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('max_epochs', 30),
                eta_min=config.get('lr_eta_min', 1e-5),
            )
        else:
            self.scheduler = None

        self.env = ILSEnvironment(config)

    def set_epoch(self, epoch: int):
        """現在のエポック番号を設定する (warm-up 判定用)。"""
        self.current_epoch = epoch

    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        1サンプルの ILS エピソードを実行し、損失を計算・更新する。

        PPO モード: 同一 trajectory に対して複数回更新。
        A2C モード: 1回更新 (従来と同一)。

        Returns:
            metrics 辞書
        """
        self.model.train()

        trajectory = self._run_ils_episode(batch_data, deterministic=False)

        if len(trajectory['rewards']) == 0:
            return self._empty_metrics()

        if self.use_ppo:
            metrics = self._ppo_update(trajectory)
        else:
            metrics = self._a2c_update(trajectory)

        return metrics

    def _a2c_update(self, trajectory: Dict) -> Dict[str, float]:
        """従来の A2C 1回更新パス。"""
        loss, loss_components = self._compute_a2c_loss(trajectory)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return self._build_train_metrics(trajectory, loss_components)

    def _ppo_update(self, trajectory: Dict) -> Dict[str, float]:
        """PPO 複数回更新パス。"""
        # old log_probs を detach
        log_probs_l1_old = torch.stack(trajectory['log_probs_l1']).detach()  # [T]
        log_probs_l2_old = torch.stack(trajectory['log_probs_l2']).detach()  # [T]

        # returns を事前計算
        rewards = trajectory['rewards']
        T = len(rewards)
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        last_loss_components = None
        first_loss_components = None

        # Warm-up 中は Critic only なので多回更新不要
        update_epochs = 1 if self.current_epoch < self.warmup_epochs else self.ppo_update_epochs

        for ppo_epoch in range(update_epochs):
            # 現在のポリシーで re-forward
            (log_probs_l1_new, log_probs_l2_new,
             entropies_l1, entropies_l2, state_values) = self._recompute_log_probs_and_values(trajectory)

            loss, loss_components = self._compute_ppo_loss(
                log_probs_l1_old, log_probs_l2_old,
                log_probs_l1_new, log_probs_l2_new,
                entropies_l1, entropies_l2,
                state_values, returns_t,
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            if first_loss_components is None:
                first_loss_components = loss_components
            last_loss_components = loss_components

        # 初回と最終の ratio を比較できるようにする
        if update_epochs > 1 and first_loss_components is not None:
            last_loss_components['ratio_l1_mean_first'] = first_loss_components.get('ratio_l1_mean', 0.0)
            last_loss_components['ratio_l2_mean_first'] = first_loss_components.get('ratio_l2_mean', 0.0)

        return self._build_train_metrics(trajectory, last_loss_components)

    def _build_train_metrics(self, trajectory: Dict, loss_components: Dict[str, float]) -> Dict[str, float]:
        """共通メトリクス構築。"""
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

        PPO モードでは各ステップの中間状態も保存する。

        Returns:
            trajectory 辞書 (log_probs, values, rewards, states など)
        """
        state = self.env.reset(
            G=batch_data['graph'],
            commodity_list=batch_data['commodity_list'],
            x_nodes=batch_data['x_nodes'],
            x_commodities=batch_data['x_commodities'],
            x_edges_capacity=batch_data['x_edges_capacity'],
        )

        trajectory: Dict[str, Any] = {
            'log_probs_l1': [],
            'log_probs_l2': [],
            'entropies_l1': [],
            'entropies_l2': [],
            'state_values': [],
            'rewards': [],
            'initial_load_factor': state['load_factor'],
            'final_load_factor': state['load_factor'],
        }

        # PPO: 不変入力と各ステップの状態を保存
        if self.use_ppo and not deterministic:
            trajectory['static_inputs'] = {
                'x_nodes': batch_data['x_nodes'].detach().clone(),
                'x_commodities': batch_data['x_commodities'].detach().clone(),
                'x_edges_capacity': batch_data['x_edges_capacity'].detach().clone(),
            }
            trajectory['states'] = []

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

            # PPO: 中間状態を保存
            if self.use_ppo and not deterministic:
                step_state = {
                    'x_edges_usage': x_edges_usage.detach().clone(),
                    'commodity_mask': commodity_mask.detach().clone(),
                    'current_assignment': copy.deepcopy(state['current_assignment']),
                    'demands': demands.detach().clone(),
                    'selected_commodity_idx': c_idx,
                    'path_mask': path_mask.detach().clone(),
                    'candidate_paths': copy.deepcopy(candidate_paths),
                    'current_paths': copy.deepcopy(current_paths_batch),
                    'demand_c': demand_c.detach().clone(),
                    'action_l1': selected_commodity.detach().clone(),  # [1]
                    'action_l2': selected_path_idx.detach().clone(),   # [1]
                }
                trajectory['states'].append(step_state)

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

    def _recompute_log_probs_and_values(
        self, trajectory: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        保存された trajectory の各ステップを現在のポリシーで re-forward し、
        同じアクションに対する log_prob と value を再計算する。

        Returns:
            (log_probs_l1_new, log_probs_l2_new, entropies_l1, entropies_l2, state_values)
            全て [T]
        """
        static = trajectory['static_inputs']
        x_nodes = static['x_nodes'].to(self.device)
        x_commodities = static['x_commodities'].to(self.device)
        x_edges_capacity = static['x_edges_capacity'].to(self.device)

        log_probs_l1_list = []
        log_probs_l2_list = []
        entropies_l1_list = []
        entropies_l2_list = []
        state_values_list = []

        for step_state in trajectory['states']:
            x_edges_usage = step_state['x_edges_usage'].to(self.device)
            commodity_mask = step_state['commodity_mask'].to(self.device)
            demands = step_state['demands'].to(self.device)

            # GNN エンコード
            node_features, edge_features, graph_embedding = self.model.encode(
                x_nodes, x_commodities, x_edges_capacity, x_edges_usage
            )

            # Critic
            state_value = self.model.get_value(node_features, graph_embedding)

            # Level1: 保存されたアクションの log_prob を取得
            current_assignment_batch = [step_state['current_assignment']]
            path_commodity_features = self.model._build_commodity_path_features(
                edge_features, current_assignment_batch, demands
            )
            action_probs_l1, log_probs_all_l1, entropy_l1 = self.model.commodity_selector(
                path_commodity_features, commodity_mask
            )
            action_l1 = step_state['action_l1'].to(self.device)  # [1]
            log_prob_l1 = log_probs_all_l1.gather(1, action_l1.unsqueeze(1)).squeeze(1)  # [1]

            # Level2: 保存されたアクションの log_prob を取得
            path_mask = step_state['path_mask'].to(self.device)
            candidate_paths = step_state['candidate_paths']
            current_paths = step_state['current_paths']
            demand_c = step_state['demand_c'].to(self.device)
            demands_norm = demand_c / self.model.demand_max
            action_probs_l2, log_probs_all_l2, entropy_l2 = self.model.path_selector(
                edge_features, action_l1, candidate_paths, current_paths, demands_norm, path_mask
            )
            action_l2 = step_state['action_l2'].to(self.device)  # [1]
            log_prob_l2 = log_probs_all_l2.gather(1, action_l2.unsqueeze(1)).squeeze(1)  # [1]

            log_probs_l1_list.append(log_prob_l1[0])
            log_probs_l2_list.append(log_prob_l2[0])
            entropies_l1_list.append(entropy_l1[0])
            entropies_l2_list.append(entropy_l2[0])
            state_values_list.append(state_value[0])

        return (
            torch.stack(log_probs_l1_list),
            torch.stack(log_probs_l2_list),
            torch.stack(entropies_l1_list),
            torch.stack(entropies_l2_list),
            torch.stack(state_values_list),
        )

    def _compute_ppo_loss(
        self,
        log_probs_l1_old: torch.Tensor,  # [T] detached
        log_probs_l2_old: torch.Tensor,  # [T] detached
        log_probs_l1_new: torch.Tensor,  # [T]
        log_probs_l2_new: torch.Tensor,  # [T]
        entropies_l1: torch.Tensor,      # [T]
        entropies_l2: torch.Tensor,      # [T]
        state_values: torch.Tensor,      # [T]
        returns_t: torch.Tensor,         # [T]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        PPO クリップ損失を計算する。

        Returns:
            (total_loss, loss_components 辞書)
        """
        T = returns_t.shape[0]
        eps = self.ppo_clip_eps
        is_warmup = self.current_epoch < self.warmup_epochs

        # Advantage
        advantages = returns_t - state_values.detach()
        adv_mean_raw = advantages.mean().item()
        adv_std_raw = advantages.std().item() if T > 1 else 0.0
        if self.normalize_advantages and T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped surrogate — Level1
        ratio_l1 = torch.exp(log_probs_l1_new - log_probs_l1_old)
        clipped_l1 = torch.clamp(ratio_l1, 1.0 - eps, 1.0 + eps)
        actor_l1_loss = -torch.min(ratio_l1 * advantages, clipped_l1 * advantages).mean()

        # PPO clipped surrogate — Level2
        ratio_l2 = torch.exp(log_probs_l2_new - log_probs_l2_old)
        clipped_l2 = torch.clamp(ratio_l2, 1.0 - eps, 1.0 + eps)
        actor_l2_loss = -torch.min(ratio_l2 * advantages, clipped_l2 * advantages).mean()

        # Critic loss
        critic_loss = F.mse_loss(state_values, returns_t)

        # Entropy
        entropy_l1_mean = entropies_l1.mean()
        entropy_l2_mean = entropies_l2.mean()

        # Warm-up: Actor 損失 + エントロピー項を無効化 (Critic only)
        if is_warmup:
            actor_l1_loss_effective = torch.tensor(0.0, device=self.device)
            actor_l2_loss_effective = torch.tensor(0.0, device=self.device)
            entropy_l1_effective = torch.tensor(0.0, device=self.device)
            entropy_l2_effective = torch.tensor(0.0, device=self.device)
        else:
            actor_l1_loss_effective = actor_l1_loss
            actor_l2_loss_effective = actor_l2_loss
            entropy_l1_effective = entropy_l1_mean
            entropy_l2_effective = entropy_l2_mean

        total_loss = (
            actor_l1_loss_effective
            + actor_l2_loss_effective
            + self.value_loss_weight * critic_loss
            - self.entropy_weight_l1 * entropy_l1_effective
            - self.entropy_weight_l2 * entropy_l2_effective
        )

        # 診断メトリクス
        with torch.no_grad():
            clip_frac_l1 = ((ratio_l1 - 1.0).abs() > eps).float().mean().item()
            clip_frac_l2 = ((ratio_l2 - 1.0).abs() > eps).float().mean().item()

        loss_components = {
            'total_loss': total_loss.item(),
            'actor_l1_loss': actor_l1_loss.item(),
            'actor_l2_loss': actor_l2_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_l1': entropy_l1_mean.item(),
            'entropy_l2': entropy_l2_mean.item(),
            'advantage_mean': adv_mean_raw,
            'advantage_std': adv_std_raw,
            'is_warmup': is_warmup,
            'ratio_l1_mean': ratio_l1.mean().item(),
            'ratio_l2_mean': ratio_l2.mean().item(),
            'ratio_l1_max': ratio_l1.max().item(),
            'ratio_l2_max': ratio_l2.max().item(),
            'clip_frac_l1': clip_frac_l1,
            'clip_frac_l2': clip_frac_l2,
        }
        return total_loss, loss_components

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
        is_warmup = self.current_epoch < self.warmup_epochs

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
        adv_mean_raw = advantages.mean().item()
        adv_std_raw  = advantages.std().item() if T > 1 else 0.0
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

        # Warm-up: Actor 損失 + エントロピー項を無効化 (Critic only)
        if is_warmup:
            actor_l1_loss_effective = torch.tensor(0.0, device=self.device)
            actor_l2_loss_effective = torch.tensor(0.0, device=self.device)
            entropy_l1_effective = torch.tensor(0.0, device=self.device)
            entropy_l2_effective = torch.tensor(0.0, device=self.device)
        else:
            actor_l1_loss_effective = actor_l1_loss
            actor_l2_loss_effective = actor_l2_loss
            entropy_l1_effective = entropy_l1_mean
            entropy_l2_effective = entropy_l2_mean

        total_loss = (
            actor_l1_loss_effective
            + actor_l2_loss_effective
            + self.value_loss_weight * critic_loss
            - self.entropy_weight_l1 * entropy_l1_effective
            - self.entropy_weight_l2 * entropy_l2_effective
        )

        l1_params = list(self.model.commodity_selector.commodity_mlp.parameters())
        l2_params = list(self.model.path_selector.path_score_mlp.parameters())

        def _grad_norm(loss, params):
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return float(sum(g.norm().item() for g in grads if g is not None))

        grad_norm_l1 = _grad_norm(actor_l1_loss, l1_params) if not is_warmup else 0.0
        grad_norm_l2 = _grad_norm(actor_l2_loss, l2_params) if not is_warmup else 0.0

        loss_components = {
            'total_loss': total_loss.item(),
            'actor_l1_loss': actor_l1_loss.item(),
            'actor_l2_loss': actor_l2_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_l1': entropy_l1_mean.item(),
            'entropy_l2': entropy_l2_mean.item(),
            'grad_norm_l1': grad_norm_l1,
            'grad_norm_l2': grad_norm_l2,
            'advantage_mean': adv_mean_raw,
            'advantage_std':  adv_std_raw,
            'is_warmup': is_warmup,
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
