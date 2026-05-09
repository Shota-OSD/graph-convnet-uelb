"""
GNN-ILS パス特徴量ベクトル化 + パスプールキャッシュの正確性テスト。

テスト対象:
  - Task 1: PathPoolManager のキャッシュ
  - Task 2a: GNNILSModel._build_commodity_path_features
  - Task 2b: PathSelector.forward + _batch_encode_paths
"""
import copy
import time

import networkx as nx
import torch
import torch.nn as nn

from src.gnn_ils.environment.path_pool_manager import PathPoolManager
from src.gnn_ils.models.gnn_ils_model import GNNILSModel
from src.gnn_ils.models.path_selector import PathSelector


# ============================================================
# 旧実装 (参照用)
# ============================================================

def _build_commodity_path_features_old(
    model, edge_features, current_assignment, demands
):
    """旧実装: for b / for c 二重ループ版。"""
    B, V, _, C, H = edge_features.shape
    device = edge_features.device

    batch_feats = []
    for b in range(B):
        commodity_feats = []
        for c in range(C):
            path = current_assignment[b][c]
            if len(path) < 2:
                commodity_feats.append(torch.zeros(H, device=device))
                continue
            edge_list = [edge_features[b, path[i], path[i + 1], c, :] for i in range(len(path) - 1)]
            stacked = torch.stack(edge_list)
            if model.path_aggregation == 'max':
                commodity_feats.append(stacked.max(dim=0).values)
            else:
                commodity_feats.append(stacked.mean(dim=0))
        batch_feats.append(torch.stack(commodity_feats))
    path_feats = torch.stack(batch_feats)

    demand_norm = (demands / model.demand_max).unsqueeze(-1)
    return torch.cat([path_feats, demand_norm], dim=-1)


def _path_selector_forward_old(selector, edge_features, selected_commodity,
                                candidate_paths, current_paths, demands, path_mask):
    """旧実装: for b / for p_idx 二重ループ版。"""
    B = edge_features.shape[0]
    neg_inf = torch.tensor(float('-inf'), device=edge_features.device)

    def _encode_path(ef, path, c_idx, b_idx):
        if len(path) < 2:
            return torch.zeros(selector.hidden_dim, device=ef.device)
        edge_feats = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_feats.append(ef[b_idx, u, v, c_idx, :])
        stacked = torch.stack(edge_feats)
        if selector.path_aggregation == 'max':
            return stacked.max(dim=0).values
        return stacked.mean(dim=0)

    import torch.nn.functional as F
    batch_scores = []
    for b in range(B):
        c_idx = selected_commodity[b].item()
        paths = candidate_paths[b]
        current_feat = _encode_path(edge_features, current_paths[b], c_idx, b)
        demand_val = demands[b].unsqueeze(0)
        path_scores = []
        for p_idx in range(selector.max_paths):
            if p_idx < len(paths):
                cand_feat = _encode_path(edge_features, paths[p_idx], c_idx, b)
                inp = torch.cat([cand_feat, current_feat, demand_val], dim=0).unsqueeze(0)
                path_scores.append(selector.path_score_mlp(inp).squeeze())
            else:
                path_scores.append(neg_inf)
        batch_scores.append(torch.stack(path_scores))
    scores = torch.stack(batch_scores)

    scores = scores.masked_fill(~path_mask, float('-inf'))
    action_probs = F.softmax(scores, dim=-1)
    log_probs = F.log_softmax(scores, dim=-1)
    log_probs_safe = torch.where(path_mask, log_probs, torch.zeros_like(log_probs))
    entropy = -(action_probs * log_probs_safe).sum(dim=-1)
    return action_probs, log_probs, entropy


# ============================================================
# テストヘルパー
# ============================================================

def _make_test_graph(num_nodes=10):
    """テスト用の有向グラフを生成する。"""
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and (i + j) % 3 != 0:
                G.add_edge(i, j, weight=1.0, capacity=100.0)
    return G


def _make_test_config(num_nodes=10, num_commodities=5, hidden_dim=32):
    return {
        'num_nodes': num_nodes,
        'num_commodities': num_commodities,
        'hidden_dim': hidden_dim,
        'num_layers': 2,
        'aggregation': 'mean',
        'encoder_dropout_rate': 0.0,
        'mlp_dropout_rate': 0.0,
        'max_candidate_paths': 8,
        'K': 5,
        'max_disjoint': 3,
        'path_aggregation': 'mean',
    }


def _make_test_data(config):
    """テスト用のデータを生成する。"""
    B = 1
    V = config['num_nodes']
    C = config['num_commodities']
    H = config['hidden_dim']
    max_paths = config['max_candidate_paths']

    edge_features = torch.randn(B, V, V, C, H, requires_grad=True)
    demands = torch.rand(B, C) * 100 + 10

    # current_assignment: ランダムなパス
    current_assignment = []
    for b in range(B):
        batch_paths = []
        for c in range(C):
            path_len = torch.randint(2, 5, (1,)).item()
            path = torch.randint(0, V, (path_len,)).tolist()
            batch_paths.append(path)
        current_assignment.append(batch_paths)

    # candidate_paths: コモディティ0のパスプール
    candidate_paths = []
    for b in range(B):
        paths = []
        num_valid = torch.randint(3, max_paths + 1, (1,)).item()
        for p in range(num_valid):
            path_len = torch.randint(2, 5, (1,)).item()
            path = torch.randint(0, V, (path_len,)).tolist()
            paths.append(path)
        candidate_paths.append(paths)

    # path_mask
    path_mask = torch.zeros(B, max_paths, dtype=torch.bool)
    for b in range(B):
        path_mask[b, :len(candidate_paths[b])] = True

    return {
        'edge_features': edge_features,
        'demands': demands,
        'current_assignment': current_assignment,
        'candidate_paths': candidate_paths,
        'path_mask': path_mask,
    }


# ============================================================
# Task 1: パスプールキャッシュのテスト
# ============================================================

def test_path_pool_cache():
    """パスプールキャッシュの動作確認。"""
    config = _make_test_config()
    manager = PathPoolManager(config)
    G = _make_test_graph(config['num_nodes'])
    commodity_list = [[0, 5, 100], [1, 7, 200], [2, 8, 150]]

    # 1回目: キャッシュミス
    t0 = time.time()
    pool1 = manager.build_path_pool(G, commodity_list)
    t1 = time.time()
    first_time = t1 - t0

    # 2回目: キャッシュヒット
    t0 = time.time()
    pool2 = manager.build_path_pool(G, commodity_list)
    t2 = time.time()
    second_time = t2 - t0

    assert pool1 == pool2, "キャッシュヒット時の結果が不一致"
    print(f"  1回目: {first_time * 1000:.2f}ms, 2回目 (cache): {second_time * 1000:.2f}ms")

    # deep copy 確認: 返り値を変更してもキャッシュに影響しない
    pool2[0][0] = [999, 888]
    pool3 = manager.build_path_pool(G, commodity_list)
    assert pool3 != pool2, "deep copy が機能していない"
    assert pool3 == pool1, "キャッシュが汚染されている"

    # 異なる commodity_list ではキャッシュミス
    commodity_list_2 = [[0, 3, 100], [1, 4, 200]]
    pool4 = manager.build_path_pool(G, commodity_list_2)
    assert len(pool4) == 2

    # clear_cache
    manager.clear_cache()
    assert len(manager._cache) == 0

    print("  PASSED: test_path_pool_cache")


# ============================================================
# Task 2a: _build_commodity_path_features のテスト
# ============================================================

def test_build_commodity_path_features_equivalence():
    """新旧実装の出力が一致することを確認。"""
    config = _make_test_config()
    model = GNNILSModel(config)
    model.eval()

    data = _make_test_data(config)
    ef = data['edge_features']
    ca = data['current_assignment']
    dm = data['demands']

    # 新実装
    new_result = model._build_commodity_path_features(ef, ca, dm)
    # 旧実装
    old_result = _build_commodity_path_features_old(model, ef, ca, dm)

    assert torch.allclose(old_result, new_result, atol=1e-6), \
        f"Max diff: {(old_result - new_result).abs().max().item()}"
    print("  PASSED: test_build_commodity_path_features_equivalence (mean)")


def test_build_commodity_path_features_max_aggregation():
    """path_aggregation='max' での新旧一致。"""
    config = _make_test_config()
    config['path_aggregation'] = 'max'
    model = GNNILSModel(config)
    model.eval()

    data = _make_test_data(config)
    ef = data['edge_features']
    ca = data['current_assignment']
    dm = data['demands']

    new_result = model._build_commodity_path_features(ef, ca, dm)
    old_result = _build_commodity_path_features_old(model, ef, ca, dm)

    assert torch.allclose(old_result, new_result, atol=1e-6), \
        f"Max diff: {(old_result - new_result).abs().max().item()}"
    print("  PASSED: test_build_commodity_path_features_max_aggregation")


def test_build_commodity_path_features_short_paths():
    """パス長1 (エッジなし) のコモディティがある場合。"""
    config = _make_test_config()
    model = GNNILSModel(config)
    model.eval()

    data = _make_test_data(config)
    ef = data['edge_features']
    ca = data['current_assignment']
    dm = data['demands']

    # コモディティ0のパスを長さ1に設定
    ca[0][0] = [3]
    # コモディティ1のパスを空に設定
    ca[0][1] = []

    new_result = model._build_commodity_path_features(ef, ca, dm)
    old_result = _build_commodity_path_features_old(model, ef, ca, dm)

    assert torch.allclose(old_result, new_result, atol=1e-6), \
        f"Max diff: {(old_result - new_result).abs().max().item()}"
    print("  PASSED: test_build_commodity_path_features_short_paths")


def test_build_commodity_path_features_gradient():
    """勾配が一致することを確認。"""
    config = _make_test_config()
    model = GNNILSModel(config)
    model.eval()

    data = _make_test_data(config)
    ca = data['current_assignment']
    dm = data['demands']

    # 旧実装
    ef1 = data['edge_features'].detach().clone().requires_grad_(True)
    old_result = _build_commodity_path_features_old(model, ef1, ca, dm)
    old_result.sum().backward()

    # 新実装
    ef2 = data['edge_features'].detach().clone().requires_grad_(True)
    new_result = model._build_commodity_path_features(ef2, ca, dm)
    new_result.sum().backward()

    assert torch.allclose(ef1.grad, ef2.grad, atol=1e-6), \
        f"Grad max diff: {(ef1.grad - ef2.grad).abs().max().item()}"
    print("  PASSED: test_build_commodity_path_features_gradient")


# ============================================================
# Task 2b: PathSelector.forward のテスト
# ============================================================

def test_path_selector_forward_equivalence():
    """PathSelector 新旧実装の出力が一致することを確認。"""
    config = _make_test_config()
    H = config['hidden_dim']
    max_paths = config['max_candidate_paths']

    selector = PathSelector(
        hidden_dim=H,
        max_paths=max_paths,
        mlp_layers=2,
        path_aggregation='mean',
        dropout_rate=0.0,
    )
    selector.eval()

    data = _make_test_data(config)
    ef = data['edge_features']
    selected_commodity = torch.tensor([0])
    candidate_paths = data['candidate_paths']
    current_paths = [data['current_assignment'][0][0]]
    demands = torch.tensor([0.5])
    path_mask = data['path_mask']

    # 新実装
    probs_new, log_new, ent_new = selector(
        ef, selected_commodity, candidate_paths, current_paths, demands, path_mask
    )
    # 旧実装
    probs_old, log_old, ent_old = _path_selector_forward_old(
        selector, ef, selected_commodity, candidate_paths, current_paths, demands, path_mask
    )

    assert torch.allclose(probs_old, probs_new, atol=1e-5), \
        f"probs max diff: {(probs_old - probs_new).abs().max().item()}"
    assert torch.allclose(log_old, log_new, atol=1e-5), \
        f"log_probs max diff: {(log_old - log_new).abs().max().item()}"
    assert torch.allclose(ent_old, ent_new, atol=1e-5), \
        f"entropy max diff: {(ent_old - ent_new).abs().max().item()}"
    print("  PASSED: test_path_selector_forward_equivalence (mean)")


def test_path_selector_forward_max_aggregation():
    """path_aggregation='max' での新旧一致。"""
    config = _make_test_config()
    H = config['hidden_dim']
    max_paths = config['max_candidate_paths']

    selector = PathSelector(
        hidden_dim=H,
        max_paths=max_paths,
        mlp_layers=2,
        path_aggregation='max',
        dropout_rate=0.0,
    )
    selector.eval()

    data = _make_test_data(config)
    ef = data['edge_features']
    selected_commodity = torch.tensor([0])
    candidate_paths = data['candidate_paths']
    current_paths = [data['current_assignment'][0][0]]
    demands = torch.tensor([0.5])
    path_mask = data['path_mask']

    probs_new, log_new, ent_new = selector(
        ef, selected_commodity, candidate_paths, current_paths, demands, path_mask
    )
    probs_old, log_old, ent_old = _path_selector_forward_old(
        selector, ef, selected_commodity, candidate_paths, current_paths, demands, path_mask
    )

    assert torch.allclose(probs_old, probs_new, atol=1e-5), \
        f"probs max diff: {(probs_old - probs_new).abs().max().item()}"
    print("  PASSED: test_path_selector_forward_max_aggregation")


def test_path_selector_gradient():
    """PathSelector の勾配が一致することを確認。"""
    config = _make_test_config()
    H = config['hidden_dim']
    max_paths = config['max_candidate_paths']

    selector = PathSelector(
        hidden_dim=H, max_paths=max_paths, mlp_layers=2,
        path_aggregation='mean', dropout_rate=0.0,
    )
    selector.eval()

    data = _make_test_data(config)
    selected_commodity = torch.tensor([0])
    candidate_paths = data['candidate_paths']
    current_paths = [data['current_assignment'][0][0]]
    demands = torch.tensor([0.5])
    path_mask = data['path_mask']

    # 旧実装
    ef1 = data['edge_features'].detach().clone().requires_grad_(True)
    probs_old, _, _ = _path_selector_forward_old(
        selector, ef1, selected_commodity, candidate_paths, current_paths, demands, path_mask
    )
    probs_old.sum().backward()

    # 新実装
    ef2 = data['edge_features'].detach().clone().requires_grad_(True)
    probs_new, _, _ = selector(
        ef2, selected_commodity, candidate_paths, current_paths, demands, path_mask
    )
    probs_new.sum().backward()

    assert torch.allclose(ef1.grad, ef2.grad, atol=1e-5), \
        f"Grad max diff: {(ef1.grad - ef2.grad).abs().max().item()}"
    print("  PASSED: test_path_selector_gradient")


# ============================================================
# 性能テスト
# ============================================================

def test_performance_comparison():
    """新旧実装の性能比較。"""
    config = _make_test_config(num_nodes=20, num_commodities=10, hidden_dim=64)
    model = GNNILSModel(config)
    model.eval()

    data = _make_test_data(config)
    ef = data['edge_features'].detach()
    ca = data['current_assignment']
    dm = data['demands']

    N = 100

    # 旧実装
    t0 = time.time()
    for _ in range(N):
        _build_commodity_path_features_old(model, ef, ca, dm)
    old_time = (time.time() - t0) / N * 1000

    # 新実装
    t0 = time.time()
    for _ in range(N):
        model._build_commodity_path_features(ef, ca, dm)
    new_time = (time.time() - t0) / N * 1000

    print(f"  _build_commodity_path_features: old={old_time:.3f}ms, new={new_time:.3f}ms")
    print(f"  PASSED: test_performance_comparison")


# ============================================================
# メインエントリポイント
# ============================================================

if __name__ == '__main__':
    torch.manual_seed(42)

    print("\n=== Task 1: PathPoolManager Cache ===")
    test_path_pool_cache()

    print("\n=== Task 2a: _build_commodity_path_features ===")
    test_build_commodity_path_features_equivalence()
    test_build_commodity_path_features_max_aggregation()
    test_build_commodity_path_features_short_paths()
    test_build_commodity_path_features_gradient()

    print("\n=== Task 2b: PathSelector.forward ===")
    test_path_selector_forward_equivalence()
    test_path_selector_forward_max_aggregation()
    test_path_selector_gradient()

    print("\n=== Performance ===")
    test_performance_comparison()

    print("\nAll tests passed!")
