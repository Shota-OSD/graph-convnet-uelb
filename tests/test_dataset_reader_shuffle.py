"""DatasetReader のシャッフル機能テスト。

テスト観点:
1. shuffle=False でエポック間の順序が同一
2. shuffle=True でエポック間の順序が変化する
3. シャッフル後もグラフ・コモディティ・厳密解の組み合わせが正しい
4. 全サンプルが欠落・重複なく使われる
"""

import csv
import os
import shutil
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.common.data_management.dataset_reader import DatasetReader


NUM_SAMPLES = 20
NUM_NODES = 4
NUM_COMMODITIES = 2


def _make_graph(idx, num_nodes):
    """サンプル idx ごとに異なる容量を持つグラフを生成。"""
    G = nx.DiGraph()
    for n in range(num_nodes):
        G.add_node(n)
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u != v:
                G.add_edge(u, v, capacity=100 * idx + u * num_nodes + v)
    return G


def _make_commodity(idx, num_nodes, num_commodities):
    """サンプル idx ごとに異なるコモディティ。demand に idx を埋め込む。"""
    commodities = []
    for c in range(num_commodities):
        src = c % num_nodes
        dst = (c + 1) % num_nodes
        demand = idx * 10 + c
        commodities.append((src, dst, demand))
    return commodities


def _make_node_flow(commodities, num_nodes):
    """最短の 2 ホップパス (src → dst) を node_flow 形式で返す。"""
    rows = []
    for src, dst, _ in commodities:
        row = [0] * num_nodes
        row[src] = 1
        row[dst] = 2
        rows.append(row)
    return rows


@pytest.fixture()
def dataset_dir():
    """テスト用ミニデータセットを一時ディレクトリに生成し、パスを返す。"""
    tmpdir = tempfile.mkdtemp()
    mode_dir = Path(tmpdir) / "train_data"

    for subdir in ("graph_file", "commodity_file", "node_flow_file"):
        (mode_dir / subdir).mkdir(parents=True)

    load_factors = []

    for idx in range(NUM_SAMPLES):
        bucket = idx - (idx % 10)
        for subdir in ("graph_file", "commodity_file", "node_flow_file"):
            (mode_dir / subdir / str(bucket)).mkdir(exist_ok=True)

        # グラフ
        G = _make_graph(idx, NUM_NODES)
        gml_path = mode_dir / "graph_file" / str(bucket) / f"graph_{idx}.gml"
        nx.write_gml(G, str(gml_path))

        # コモディティ
        commodities = _make_commodity(idx, NUM_NODES, NUM_COMMODITIES)
        csv_path = mode_dir / "commodity_file" / str(bucket) / f"commodity_data_{idx}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for src, dst, demand in commodities:
                writer.writerow([src, dst, demand])

        # ノードフロー
        node_flow = _make_node_flow(commodities, NUM_NODES)
        nf_path = mode_dir / "node_flow_file" / str(bucket) / f"node_flow_{idx}.csv"
        with open(nf_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in node_flow:
                writer.writerow(row)

        # 厳密解: idx ごとにユニークな値
        load_factors.append(1.0 + idx * 0.01)

    # exact_solution.csv
    with open(mode_dir / "exact_solution.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for lf in load_factors:
            writer.writerow([lf])

    yield tmpdir
    shutil.rmtree(tmpdir)


class _FakeConfig:
    """DatasetReader が必要とする最小限の config。"""

    def __init__(self, tmpdir):
        self._tmpdir = tmpdir

    def get(self, key, default=None):
        if key == "dataset_name":
            return None
        if key == "expt_name":
            return None
        if key == "data_root":
            return None
        return default


def _make_reader(dataset_dir, shuffle, batch_size=1):
    """get_mode_dir をバイパスして data_dir を直接セットする。"""
    config = _FakeConfig(dataset_dir)
    reader = DatasetReader.__new__(DatasetReader)
    reader.num_data = NUM_SAMPLES
    reader.batch_size = batch_size
    reader.mode = "train"
    reader.shuffle = shuffle
    reader.max_iter = NUM_SAMPLES // batch_size
    reader.data_dir = Path(dataset_dir) / "train_data"
    reader._load_factors = reader._load_exact_solutions()
    return reader


# -------------------------------------------------------------------
# テスト
# -------------------------------------------------------------------


class TestShuffleDisabled:
    """shuffle=False のとき、順序が固定されることを確認。"""

    def test_order_is_deterministic(self, dataset_dir):
        reader = _make_reader(dataset_dir, shuffle=False)
        lf_epoch1 = [b.load_factor.item() for b in reader]
        lf_epoch2 = [b.load_factor.item() for b in reader]
        assert lf_epoch1 == lf_epoch2

    def test_sequential_order(self, dataset_dir):
        reader = _make_reader(dataset_dir, shuffle=False)
        load_factors = [b.load_factor.item() for b in reader]
        expected = [1.0 + i * 0.01 for i in range(NUM_SAMPLES)]
        np.testing.assert_allclose(load_factors, expected)


class TestShuffleEnabled:
    """shuffle=True のとき、順序が変化することを確認。"""

    def test_order_changes_across_epochs(self, dataset_dir):
        reader = _make_reader(dataset_dir, shuffle=True)
        orders = []
        for _ in range(10):
            lf = [b.load_factor.item() for b in reader]
            orders.append(tuple(lf))
        unique_orders = set(orders)
        assert len(unique_orders) > 1, "10エポック全て同一順序はシャッフルされていない"

    def test_all_samples_present(self, dataset_dir):
        """シャッフル後も全サンプルが欠落・重複なく含まれる。"""
        reader = _make_reader(dataset_dir, shuffle=True)
        load_factors = sorted([b.load_factor.item() for b in reader])
        expected = sorted([1.0 + i * 0.01 for i in range(NUM_SAMPLES)])
        np.testing.assert_allclose(load_factors, expected)


class TestDataConsistency:
    """シャッフル後もグラフ・コモディティ・厳密解の組み合わせが正しいことを確認。"""

    def test_commodity_demand_matches_load_factor(self, dataset_dir):
        """各サンプルの demand と load_factor から元の idx を復元し、一致を検証。"""
        reader = _make_reader(dataset_dir, shuffle=True)
        for batch in reader:
            load_factor = batch.load_factor.item()
            # load_factor = 1.0 + idx * 0.01 → idx を逆算
            idx = round((load_factor - 1.0) / 0.01)

            # demand = idx * 10 + c で埋め込んだ値と照合
            commodities = batch.commodities[0]  # [num_commodities, 3]
            for c in range(NUM_COMMODITIES):
                expected_demand = idx * 10 + c
                assert commodities[c, 2] == expected_demand, (
                    f"idx={idx}, commodity={c}: "
                    f"expected demand={expected_demand}, got {commodities[c, 2]}"
                )

    def test_capacity_matches_load_factor(self, dataset_dir):
        """容量行列に埋め込んだ idx と load_factor の idx が一致。"""
        reader = _make_reader(dataset_dir, shuffle=True)
        for batch in reader:
            load_factor = batch.load_factor.item()
            idx = round((load_factor - 1.0) / 0.01)

            capacity = batch.edges_capacity[0]  # [V, V]
            # capacity[0, 1] = 100 * idx + 0 * V + 1
            expected = 100 * idx + 1
            assert capacity[0, 1] == expected, (
                f"idx={idx}: expected capacity[0,1]={expected}, got {capacity[0, 1]}"
            )

    def test_node_features_match_commodities(self, dataset_dir):
        """ノード特徴量の src/dst マーカーがコモディティと一致。"""
        reader = _make_reader(dataset_dir, shuffle=True)
        for batch in reader:
            nodes = batch.nodes[0]          # [V, C]
            commodities = batch.commodities[0]  # [C, 3]
            for c in range(NUM_COMMODITIES):
                src, dst = int(commodities[c, 0]), int(commodities[c, 1])
                assert nodes[src, c] == 1, f"commodity {c}: src node should be marked 1"
                assert nodes[dst, c] == 2, f"commodity {c}: dst node should be marked 2"


class TestBatchSize:
    """batch_size > 1 でもシャッフルが正しく動作。"""

    def test_batched_shuffle_all_samples_present(self, dataset_dir):
        batch_size = 5
        reader = _make_reader(dataset_dir, shuffle=True, batch_size=batch_size)
        all_lf = []
        for batch in reader:
            assert batch.load_factor.shape[0] == batch_size
            all_lf.extend(batch.load_factor.tolist())
        expected = sorted([1.0 + i * 0.01 for i in range(NUM_SAMPLES)])
        np.testing.assert_allclose(sorted(all_lf), expected)
