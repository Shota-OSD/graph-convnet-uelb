import pytest
import random
import copy

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
        # エッジ (0,1) に demand=5 が集中するようにする
        bottleneck_path = [0, 1, 4]
        env.current_assignment[0] = bottleneck_path

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

        # perturbation 実行
        rerouted = env.perturbation_congestion_aware()

        assert rerouted, "perturbation_prob=1.0 でリルートが発生しなかった"

        # ボトルネックリンクを通っていた commodity のパスが変更されたことを確認
        changed = False
        for c_idx in range(len(env.current_assignment)):
            if env.current_assignment[c_idx] != assignment_before[c_idx]:
                changed = True
                # 新しいパスがボトルネックリンクを含まないことを確認
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
        perturbation_prob=0.0 の場合、全 commodity がスキップされるが、
        フォールバックで少なくとも1本は変更されることを確認する。
        """
        random.seed(42)
        np.random.seed(42)

        config = _make_config(perturbation_prob=0.0)
        env, _ = _setup_env(config=config)

        assignment_before = [list(p) for p in env.current_assignment]

        rerouted = env.perturbation_congestion_aware()

        assert rerouted, "フォールバックでリルートが発生しなかった"

        # 少なくとも1箇所異なることを確認
        any_changed = any(
            env.current_assignment[c] != assignment_before[c]
            for c in range(len(env.current_assignment))
        )
        assert any_changed, "perturbation_prob=0.0 でもフォールバックにより変更が発生するべき"

    def test_perturbation_guarantees_at_least_one_change(self):
        """
        任意の設定で perturbation_congestion_aware() を10回呼び、
        毎回少なくとも1つの commodity のパスが変わることを確認する。
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
            assert any_changed, (
                f"trial {trial}: perturbation 後にパスが1つも変わっていない"
            )

    def test_perturbation_does_not_select_current_path(self):
        """
        perturbation_prob=1.0 で代替パスがある commodity について、
        perturbation 後のパスが perturbation 前と同じでないことを確認する。
        """
        random.seed(42)
        np.random.seed(42)

        config = _make_config(perturbation_prob=1.0)
        env, _ = _setup_env(config=config)

        # ボトルネックを手動で作成: commodity 0 を 0->1->4 に設定
        # commodity 1 も 0->1->3 にして (0,1) に負荷集中
        env.current_assignment[0] = [0, 1, 4]
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

        assignment_before = [tuple(p) for p in env.current_assignment]

        env.perturbation_congestion_aware()

        # パスが変わった commodity について、変更後のパスが変更前と異なることを確認
        # (perturbation はボトルネックを回避する代替パスがある commodity のみ変更する)
        for c_idx in range(len(env.current_assignment)):
            current_tuple = tuple(env.current_assignment[c_idx])
            if current_tuple != assignment_before[c_idx]:
                # 変更された commodity のパスは、元のパスと異なるはず (tautology だが明示)
                assert current_tuple != assignment_before[c_idx]
                # さらに、新しいパスがボトルネックリンクを含まないことを確認
                new_path = env.current_assignment[c_idx]
                for i in range(len(new_path) - 1):
                    edge = (new_path[i], new_path[i + 1])
                    reverse_edge = (new_path[i + 1], new_path[i])
                    assert edge != (bn_u, bn_v) and reverse_edge != (bn_u, bn_v), (
                        f"commodity {c_idx}: リルート後のパスにボトルネックリンク "
                        f"({bn_u},{bn_v}) が含まれている: {new_path}"
                    )

        # 少なくとも1つは変更されているはず (prob=1.0)
        any_changed = any(
            tuple(env.current_assignment[c]) != assignment_before[c]
            for c in range(len(env.current_assignment))
        )
        assert any_changed, (
            "perturbation_prob=1.0 で1つも commodity のパスが変わらなかった"
        )


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
        """
        random.seed(42)
        config = _make_config(no_improve_patience=10, max_iterations=50)
        env, _ = _setup_env(config=config)

        # patience // 2 = 5 が閾値
        env.no_improve_count = 4
        assert env.should_perturb() is False, "no_improve_count=4 で should_perturb() が True になった"

        env.no_improve_count = 5
        assert env.should_perturb() is True, "no_improve_count=5 で should_perturb() が False になった"

        # iteration が max_iterations に達している場合は False
        env.no_improve_count = 5
        env.iteration = env.max_iterations
        assert env.should_perturb() is False, "iteration=max_iterations で should_perturb() が True になった"

    def test_apply_perturbation_resets_counter(self):
        """
        apply_perturbation() が no_improve_count を 0 にリセットすることを確認する。
        """
        random.seed(42)
        env, _ = _setup_env()

        env.no_improve_count = 5
        assert env.no_improve_count == 5

        env.apply_perturbation()

        assert env.no_improve_count == 0, (
            f"apply_perturbation() 後に no_improve_count が 0 にリセットされなかった: "
            f"{env.no_improve_count}"
        )

    def test_apply_perturbation_preserves_best(self):
        """
        apply_perturbation() が best_load_factor を変えないことを確認する。
        """
        random.seed(42)
        env, _ = _setup_env()

        best_lf_before = env.best_load_factor
        best_assignment_before = [list(p) for p in env.best_assignment]

        env.apply_perturbation()

        assert env.best_load_factor == best_lf_before, (
            f"apply_perturbation() が best_load_factor を変更した: "
            f"{best_lf_before} -> {env.best_load_factor}"
        )
        assert env.best_assignment == best_assignment_before, (
            "apply_perturbation() が best_assignment を変更した"
        )

    def test_perturbation_updates_internal_state(self):
        """
        apply_perturbation() の前後で x_edges_usage と current_load_factor が
        再計算されていることを確認する。
        """
        random.seed(42)
        env, _ = _setup_env()

        usage_before = env.x_edges_usage.clone()
        lf_before = env.current_load_factor

        env.apply_perturbation()

        # パスが変わっているなら usage も変わっているはず
        # (フォールバック含めて必ず1本は変わるので usage は変わる)
        usage_after = env.x_edges_usage.clone()
        lf_after = env.current_load_factor

        # 再計算の整合性チェック: 現在の assignment から期待される usage/lf を計算
        expected_usage = compute_edge_usage(
            env.current_assignment, env.commodity_list, env.num_nodes
        )
        expected_lf = compute_load_factor(expected_usage, env.capacity_np)

        np.testing.assert_allclose(
            usage_after[0].numpy(), expected_usage,
            rtol=1e-5,
            err_msg="x_edges_usage が current_assignment と整合していない",
        )
        assert abs(lf_after - expected_lf) < 1e-8, (
            f"current_load_factor が再計算値と一致しない: {lf_after} vs {expected_lf}"
        )
