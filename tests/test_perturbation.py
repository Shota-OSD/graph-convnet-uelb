import pytest
import random

import networkx as nx
import numpy as np
import torch

from src.gnn_ils.environment.ils_environment import ILSEnvironment
from src.gnn_ils.utils.load_utils import compute_edge_usage, compute_load_factor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_graph():
    """
    テスト用の小さな有向グラフを作成。

    ノード: 0, 1, 2, 3, 4
    エッジ（双方向、各容量10）:
        0-1, 0-2, 1-3, 2-3, 3-4, 1-4, 2-4

    これにより複数の代替パスが存在する。
    """
    G = nx.DiGraph()
    edges = [
        (0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1),
        (2, 3), (3, 2), (3, 4), (4, 3), (1, 4), (4, 1),
        (2, 4), (4, 2),
    ]
    for u, v in edges:
        G.add_edge(u, v, weight=1.0, capacity=10.0)
    return G


def _make_config(**overrides):
    config = {
        'num_nodes': 5,
        'num_commodities': 3,
        'max_iterations': 50,
        'no_improve_patience': 10,
        'perturbation_prob': 0.3,
        'max_perturbations': 3,
        'K': 5,
        'max_disjoint': 3,
        'max_candidate_paths': 10,
        'reward_mode': 'shared',
        'reward_scale': 100.0,
    }
    config.update(overrides)
    return config


def _setup_env(config=None, commodities=None):
    """
    ILSEnvironment をセットアップし、reset 済みの状態を返す。
    """
    if config is None:
        config = _make_config()

    G = _make_simple_graph()
    V = config['num_nodes']
    C = config['num_commodities']

    if commodities is None:
        commodities = [[0, 4, 5], [0, 3, 3], [1, 4, 4]]  # [src, dst, demand]

    env = ILSEnvironment(config)

    x_nodes = torch.zeros(1, V, C, dtype=torch.long)
    x_commodities = torch.tensor([commodities], dtype=torch.float32)  # [1, C, 3]
    x_edges_capacity = torch.zeros(1, V, V, dtype=torch.float32)
    for u, v, data in G.edges(data=True):
        x_edges_capacity[0, u, v] = data['capacity']

    state = env.reset(G, commodities, x_nodes, x_commodities, x_edges_capacity)
    return env, state


# ---------------------------------------------------------------------------
# 単体テスト: perturbation_congestion_aware
# ---------------------------------------------------------------------------

class TestPerturbationCongestionAware:
    """perturbation_congestion_aware() の単体テスト。"""

    def test_perturbation_reroutes_bottleneck_commodity(self):
        """
        ボトルネックリンクを通る commodity が代替パスにリルートされ、
        新しいパスはボトルネックリンクを含まないことを確認する。
        """
        random.seed(42)
        np.random.seed(42)

        config = _make_config(perturbation_prob=1.0)
        env, _ = _setup_env(config=config)

        # ボトルネックを手動で作成: commodity 0 を 0->1->4 に設定
        env.current_assignment[0] = [0, 1, 4]
        # commodity 1 も 0->1->3 を通らせて (0,1) に更に負荷をかける
        env.current_assignment[1] = [0, 1, 3]

        # 再計算
        usage_np = compute_edge_usage(
            env.current_assignment, env.commodity_list, env.num_nodes
        )
        env.x_edges_usage = torch.tensor(usage_np, dtype=torch.float32).unsqueeze(0)
        env.current_load_factor = compute_load_factor(usage_np, env.capacity_np)

        # ボトルネックリンクを特定
        with np.errstate(divide='ignore', invalid='ignore'):
            load_ratio = np.where(
                env.capacity_np > 0,
                usage_np / env.capacity_np,
                0.0,
            )
        bn_idx = np.unravel_index(np.argmax(load_ratio), load_ratio.shape)
        bn_u, bn_v = int(bn_idx[0]), int(bn_idx[1])

        assignment_before = [list(p) for p in env.current_assignment]

        rerouted = env.perturbation_congestion_aware()

        assert rerouted, "perturbation_prob=1.0 でリルートが発生しなかった"

        # ボトルネックリンクを通っていた commodity のパスが変更されたことを確認
        changed = False
        for c_idx in range(len(env.current_assignment)):
            if env.current_assignment[c_idx] != assignment_before[c_idx]:
                changed = True
                new_path = env.current_assignment[c_idx]
                for i in range(len(new_path) - 1):
                    edge = (new_path[i], new_path[i + 1])
                    reverse_edge = (new_path[i + 1], new_path[i])
                    assert edge != (bn_u, bn_v) and reverse_edge != (bn_u, bn_v), (
                        f"リルート後のパスにボトルネックリンク ({bn_u},{bn_v}) が含まれている: {new_path}"
                    )

        assert changed, "ボトルネックリンクを通る commodity のパスが変更されなかった"

    def test_perturbation_prob_zero_triggers_fallback(self):
        """
        perturbation_prob=0.0 の場合、フォールバックで少なくとも1本は変更される。
        """
        random.seed(42)
        np.random.seed(42)

        config = _make_config(perturbation_prob=0.0)
        env, _ = _setup_env(config=config)

        assignment_before = [list(p) for p in env.current_assignment]
        rerouted = env.perturbation_congestion_aware()

        assert rerouted, "フォールバックでリルートが発生しなかった"
        any_changed = any(
            env.current_assignment[c] != assignment_before[c]
            for c in range(len(env.current_assignment))
        )
        assert any_changed, "perturbation_prob=0.0 でもフォールバックにより変更が発生するべき"

    def test_perturbation_guarantees_at_least_one_change(self):
        """
        10回呼び、毎回少なくとも1つの commodity のパスが変わることを確認する。
        """
        for trial in range(10):
            random.seed(trial)
            np.random.seed(trial)

            env, _ = _setup_env()
            assignment_before = [list(p) for p in env.current_assignment]

            rerouted = env.perturbation_congestion_aware()

            assert rerouted, f"trial {trial}: perturbation がリルートを返さなかった"
            any_changed = any(
                env.current_assignment[c] != assignment_before[c]
                for c in range(len(env.current_assignment))
            )
            assert any_changed, f"trial {trial}: perturbation 後にパスが1つも変わっていない"

    def test_perturbation_does_not_select_current_path(self):
        """
        perturbation_prob=1.0 で変更された commodity のパスがボトルネックを含まないことを確認する。
        """
        random.seed(42)
        np.random.seed(42)

        config = _make_config(perturbation_prob=1.0)
        env, _ = _setup_env(config=config)

        env.current_assignment[0] = [0, 1, 4]
        env.current_assignment[1] = [0, 1, 3]

        usage_np = compute_edge_usage(
            env.current_assignment, env.commodity_list, env.num_nodes
        )
        env.x_edges_usage = torch.tensor(usage_np, dtype=torch.float32).unsqueeze(0)
        env.current_load_factor = compute_load_factor(usage_np, env.capacity_np)

        with np.errstate(divide='ignore', invalid='ignore'):
            load_ratio = np.where(env.capacity_np > 0, usage_np / env.capacity_np, 0.0)
        bn_idx = np.unravel_index(np.argmax(load_ratio), load_ratio.shape)
        bn_u, bn_v = int(bn_idx[0]), int(bn_idx[1])

        assignment_before = [tuple(p) for p in env.current_assignment]
        env.perturbation_congestion_aware()

        for c_idx in range(len(env.current_assignment)):
            current_tuple = tuple(env.current_assignment[c_idx])
            if current_tuple != assignment_before[c_idx]:
                new_path = env.current_assignment[c_idx]
                for i in range(len(new_path) - 1):
                    edge = (new_path[i], new_path[i + 1])
                    reverse_edge = (new_path[i + 1], new_path[i])
                    assert edge != (bn_u, bn_v) and reverse_edge != (bn_u, bn_v), (
                        f"commodity {c_idx}: リルート後のパスにボトルネックリンク "
                        f"({bn_u},{bn_v}) が含まれている: {new_path}"
                    )

        any_changed = any(
            tuple(env.current_assignment[c]) != assignment_before[c]
            for c in range(len(env.current_assignment))
        )
        assert any_changed, "perturbation_prob=1.0 で1つも commodity のパスが変わらなかった"

    def test_perturbation_no_alternatives_fallback(self):
        """
        全対象 commodity で代替パスが構造的に存在しない場合、
        フォールバックでボトルネックと無関係な commodity がリルートされることを確認する。
        """
        random.seed(42)

        G = nx.DiGraph()
        G.add_edge(0, 1, weight=1.0, capacity=5.0)
        G.add_edge(1, 2, weight=1.0, capacity=5.0)
        G.add_edge(0, 2, weight=1.0, capacity=10.0)
        G.add_edge(2, 0, weight=1.0, capacity=10.0)
        G.add_edge(1, 0, weight=1.0, capacity=10.0)
        G.add_edge(2, 1, weight=1.0, capacity=10.0)

        commodities = [[0, 2, 10], [0, 1, 1]]
        config = _make_config(
            num_nodes=3, num_commodities=2,
            perturbation_prob=1.0, K=3, max_disjoint=2, max_candidate_paths=5,
        )
        env = ILSEnvironment(config)

        V, C = 3, 2
        x_nodes = torch.zeros(1, V, C, dtype=torch.long)
        x_commodities = torch.tensor([commodities], dtype=torch.float32)
        x_edges_capacity = torch.zeros(1, V, V, dtype=torch.float32)
        for u, v, data in G.edges(data=True):
            x_edges_capacity[0, u, v] = data['capacity']

        env.reset(G, commodities, x_nodes, x_commodities, x_edges_capacity)

        assignment_before = [list(p) for p in env.current_assignment]
        rerouted = env.perturbation_congestion_aware()

        assert rerouted, "フォールバックでリルートが発生しなかった"
        any_changed = any(
            env.current_assignment[c] != assignment_before[c]
            for c in range(len(env.current_assignment))
        )
        assert any_changed, "代替パスなしケースでフォールバックが動作しなかった"


# ---------------------------------------------------------------------------
# 統合テスト: ILS ループ (should_perturb / apply_perturbation)
# ---------------------------------------------------------------------------

class TestILSLoopIntegration:
    """should_perturb() / apply_perturbation() の統合テスト。"""

    def test_should_perturb_trigger(self):
        """
        no_improve_patience=10 のとき:
        - no_improve_count=4 -> False
        - no_improve_count=5 -> True
        - iteration=max_iterations -> False (上限到達)
        - perturbation_count=max_perturbations -> False (上限到達)
        """
        random.seed(42)
        config = _make_config(no_improve_patience=10, max_iterations=50, max_perturbations=3)
        env, _ = _setup_env(config=config)

        env.no_improve_count = 4
        assert env.should_perturb() is False

        env.no_improve_count = 5
        assert env.should_perturb() is True

        env.no_improve_count = 5
        env.iteration = env.max_iterations
        assert env.should_perturb() is False

        env.iteration = 10
        env.no_improve_count = 5
        env.perturbation_count = 3
        assert env.should_perturb() is False

    def test_apply_perturbation_resets_counter(self):
        """apply_perturbation() が no_improve_count を 0 にリセットする。"""
        random.seed(42)
        env, _ = _setup_env()

        env.no_improve_count = 5
        env.apply_perturbation()

        assert env.no_improve_count == 0

    def test_apply_perturbation_preserves_best(self):
        """apply_perturbation() が best_load_factor を変えない。"""
        random.seed(42)
        env, _ = _setup_env()

        best_lf_before = env.best_load_factor
        best_assignment_before = [list(p) for p in env.best_assignment]

        env.apply_perturbation()

        assert env.best_load_factor == best_lf_before
        assert env.best_assignment == best_assignment_before

    def test_perturbation_updates_internal_state(self):
        """apply_perturbation() 後に内部状態が assignment と整合する。"""
        random.seed(42)
        env, _ = _setup_env()

        env.apply_perturbation()

        expected_usage = compute_edge_usage(
            env.current_assignment, env.commodity_list, env.num_nodes
        )
        expected_lf = compute_load_factor(expected_usage, env.capacity_np)

        np.testing.assert_allclose(
            env.x_edges_usage[0].numpy(), expected_usage, rtol=1e-5,
        )
        assert abs(env.current_load_factor - expected_lf) < 1e-8

    def test_max_perturbations_restores_patience_termination(self):
        """
        max_perturbations 回消費後は should_perturb() が False になり、
        patience による早期終了が有効に戻る。
        """
        random.seed(42)
        config = _make_config(
            no_improve_patience=10, max_iterations=100, max_perturbations=2
        )
        env, _ = _setup_env(config=config)

        for _ in range(2):
            env.no_improve_count = 5
            assert env.should_perturb() is True
            env.apply_perturbation()

        assert env.perturbation_count == 2
        env.no_improve_count = 5
        assert env.should_perturb() is False

        # patience=10 の done 条件に到達可能
        env.no_improve_count = 10
        assert env._check_done() is True

    def test_apply_perturbation_returns_state(self):
        """apply_perturbation() が状態辞書を返す。"""
        random.seed(42)
        env, _ = _setup_env()

        state = env.apply_perturbation()

        assert isinstance(state, dict)
        assert 'x_edges_usage' in state
        assert 'load_factor' in state
        assert 'current_assignment' in state
