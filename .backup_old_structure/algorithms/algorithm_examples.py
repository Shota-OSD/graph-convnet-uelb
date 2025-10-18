"""
ビームサーチアルゴリズムの学習・例示用ファイル
カスタムアルゴリズムの作成方法やサンプルコードを提供
"""

import torch
import os
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional
from beamsearch_uelb import BeamSearchAlgorithm
from ..data_management.dataset_reader import DatasetReader
from ..models.gcn_model import ResidualGatedGCNModel
from ..config.config_manager import ConfigManager

def load_example_config(config_path: str) -> Dict[str, Any]:
    """アルゴリズム例用の設定を読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_model_path(model_config: Dict[str, Any]) -> Optional[str]:
    """設定に基づいてモデルパスを決定"""
    model_dir = model_config.get('model_path', './saved_models')
    
    if not os.path.exists(model_dir):
        return None
    
    # 特定のモデルハッシュが指定されている場合
    if model_config.get('model_hash'):
        model_hash = model_config['model_hash']
        if model_config.get('model_epoch'):
            filename = f"model_{model_hash}_epoch_{model_config['model_epoch']}.pt"
        else:
            filename = f"model_{model_hash}_latest.pt"
        
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            return model_path
    
    # 最新のモデルを使用
    if model_config.get('use_latest_model', True):
        model_files = [f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('_latest.pt')]
        if model_files:
            return os.path.join(model_dir, model_files[0])
    
    return None

def create_sample_data_from_config(example_config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """設定ファイルに基づいてサンプルデータを作成"""
    model_config = example_config.get('model_config', {})
    data_config = example_config.get('data_config', {})
    
    config_path = model_config.get('config_path')
    model_path = find_model_path(model_config)
    batch_size = data_config.get('batch_size', 2)
    
    if config_path and model_path and os.path.exists(config_path):
        print(f"使用するモデル: {model_path}")
        print(f"使用する設定: {config_path}")
        return create_real_prediction_data(batch_size, config_path, model_path)
    else:
        # フォールバック：ランダムデータを生成
        print("設定ファイルまたはモデルファイルが見つからないため、ランダムデータを使用します")
        fallback_nodes = data_config.get('fallback_num_nodes', 10)
        fallback_commodities = data_config.get('fallback_num_commodities', 5)
        return create_random_sample_data(batch_size, fallback_nodes, fallback_commodities)

def create_sample_data_from_saved(batch_size=2, num_nodes=10, num_commodities=5, config_path=None, model_path=None):
    """保存されたデータとモデルを使用してサンプルデータを作成（後方互換性用）"""
    
    if config_path and model_path and os.path.exists(config_path) and os.path.exists(model_path):
        return create_real_prediction_data(batch_size, config_path, model_path)
    else:
        # フォールバック：ランダムデータを生成
        print("設定ファイルまたはモデルファイルが見つからないため、ランダムデータを使用します")
        return create_random_sample_data(batch_size, num_nodes, num_commodities)

def create_real_prediction_data(batch_size, config_path, model_path):
    """実際の保存済みモデルを使用して予測エッジテンソルを生成"""
    
    # 設定の読み込み
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    dtypeFloat, dtypeLong = config_manager.get_dtypes()
    
    # モデルの読み込み
    model = ResidualGatedGCNModel(config, dtypeFloat, dtypeLong)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # テストデータの読み込み
    num_test_data = min(config.get('num_test_data', 20), batch_size * 10)  # 十分なデータを確保
    dataset = DatasetReader(num_test_data, batch_size, 'test')
    
    try:
        # 最初のバッチを取得
        batch = next(iter(dataset))
        
        # テンソルに変換
        x_edges = torch.LongTensor(batch.edges).contiguous()
        x_edges_capacity = torch.FloatTensor(batch.edges_capacity).contiguous()
        x_nodes = torch.LongTensor(batch.nodes).contiguous()
        y_edges = torch.LongTensor(batch.edges_target).contiguous()
        batch_commodities = torch.LongTensor(batch.commodities).contiguous()
        x_commodities = batch_commodities[:, :, 2].to(torch.float)
        
        # モデルによる予測（推論モード）
        with torch.no_grad():
            # edge_cwはダミー値を使用（推論時は不要）
            edge_cw = torch.ones(2)
            y_preds, _ = model.forward(x_edges, x_commodities, x_edges_capacity, x_nodes, y_edges, edge_cw)
        
        print(f"実際のモデルから予測データを生成しました")
        print(f"バッチサイズ: {y_preds.shape[0]}, ノード数: {y_preds.shape[1]}, コモディティ数: {batch_commodities.shape[1]}")
        
        return y_preds, batch.edges_capacity, batch.commodities
        
    except Exception as e:
        print(f"実際のデータ読み込みに失敗: {e}")
        # フォールバック
        return create_random_sample_data(batch_size, config.get('num_nodes', 10), config.get('num_commodities', 5))

def create_random_sample_data(batch_size=2, num_nodes=10, num_commodities=5):
    """ランダムなサンプルデータの作成（フォールバック用）"""
    
    # 予測エッジテンソル (batch_size, num_nodes, num_nodes, num_commodities, num_vec)
    y_pred_edges = torch.randn(batch_size, num_nodes, num_nodes, num_commodities, 2)
    
    # エッジ容量テンソル (batch_size, num_nodes, num_nodes)
    edges_capacity = torch.randint(0, 10, (batch_size, num_nodes, num_nodes))
    # 対角成分を0に設定（自己ループを防ぐ）
    for i in range(num_nodes):
        edges_capacity[:, i, i] = 0
    
    # コモディティテンソル (batch_size, num_commodities, 3) -> (source, target, demand)
    commodities = torch.zeros(batch_size, num_commodities, 3, dtype=torch.long)
    for b in range(batch_size):
        for c in range(num_commodities):
            source = torch.randint(0, num_nodes, (1,))
            target = torch.randint(0, num_nodes, (1,))
            while target == source:  # 同じノードを避ける
                target = torch.randint(0, num_nodes, (1,))
            demand = torch.randint(1, 5, (1,))
            commodities[b, c] = torch.cat([source, target, demand])
    
    return y_pred_edges, edges_capacity, commodities

def create_sample_data(batch_size=2, num_nodes=10, num_commodities=5):
    """後方互換性のためのラッパー関数"""
    # デフォルトのモデルと設定を探す
    config_path = 'configs/load_saved_model.json'
    model_dir = './saved_models'
    
    model_path = None
    if os.path.exists(model_dir):
        # 最新のモデルファイルを探す
        model_files = [f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('_latest.pt')]
        if model_files:
            model_path = os.path.join(model_dir, model_files[0])
    
    return create_sample_data_from_saved(batch_size, num_nodes, num_commodities, config_path, model_path)

class DemandPriorityBeamSearch(BeamSearchAlgorithm):
    """需要量優先のカスタムビームサーチアルゴリズム"""
    
    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """需要量優先のビームサーチによる単一バッチ検索"""
        batch_y_pred_edges = self.y[batch]
        commodities = self.commodities[batch]
        count = 0

        while count < self.max_iter:
            batch_edges_capacity = self.edges_capacity[batch]
            
            # 需要量に基づく優先度付きソート
            demands = commodities[:, 2]
            sorted_indices = torch.argsort(demands, descending=True)
            sorted_commodities = commodities[sorted_indices]
            sorted_pred_edges = batch_y_pred_edges[:, :, sorted_indices]
            _, original_indices = torch.sort(sorted_indices)

            node_orders = []
            commodity_paths = []
            is_feasible = True

            for index, commodity in enumerate(sorted_commodities):
                source_node = commodity[0].item()
                target_node = commodity[1].item()
                demand = commodity[2].item()
                
                # ビームサーチによるパス探索
                node_order, remaining_edges_capacity, best_path = self._beam_search_for_commodity(
                    batch_edges_capacity, sorted_pred_edges[:, :, index], 
                    source_node, target_node, demand
                )
                
                if best_path == []:
                    break
                    
                node_orders.append(node_order)
                if self.mode_strict:
                    batch_edges_capacity = remaining_edges_capacity
                commodity_paths.append(best_path)

            if len(commodity_paths) == commodities.shape[0]:
                count = self.max_iter
                unshuffled_commodity_paths = [commodity_paths[i] for i in original_indices]
            else:
                is_feasible = False
                unshuffled_commodity_paths = self._get_fallback_paths(commodities.shape[0])
                count += 1
        
        return node_orders, unshuffled_commodity_paths, is_feasible

    def _beam_search_for_commodity(self, edges_capacity, y_commodities, source, target, demand):
        """標準的なビームサーチ（元の実装と同じ）"""
        beam_queue = [(source, [source], 0, edges_capacity.clone())]
        best_paths = []

        while beam_queue:
            beam_queue = sorted(beam_queue, key=lambda x: x[2], reverse=True)[:self.beam_size]

            next_beam_queue = []
            for current_node, path, current_score, remaining_edges_capacity in beam_queue:
                if current_node == target:
                    best_paths.append((path, current_score, remaining_edges_capacity))
                    continue

                for next_node in range(edges_capacity.shape[0]):
                    if next_node in path:
                        continue
                    if (edges_capacity[current_node, next_node].item() == 0):
                        continue

                    flow_probability = y_commodities[current_node, next_node]
                    new_score = current_score + flow_probability

                    if demand <= remaining_edges_capacity[current_node, next_node]:
                        updated_capacity = remaining_edges_capacity.clone()
                        updated_capacity[current_node, next_node] -= demand
                        new_path = path + [next_node]
                        next_beam_queue.append((next_node, new_path, new_score, updated_capacity))
                        
            beam_queue = next_beam_queue

        if not best_paths:
            node_order = [0] * edges_capacity.shape[0]
            return node_order, edges_capacity, []

        best_paths = sorted(best_paths, key=lambda x: x[1], reverse=True)
        node_order = [0] * edges_capacity.shape[0]
        for idx, node in enumerate(best_paths[0][0]):
            node_order[node] = idx + 1
        return node_order, best_paths[0][2], best_paths[0][0]

    def _get_fallback_paths(self, num_commodities: int) -> List[List[int]]:
        """フォールバックパスの生成"""
        return [[0,1,2,3,4,5,6,7,8,9] for _ in range(num_commodities)]

def example_custom_algorithm():
    """カスタムアルゴリズムの作成例"""
    print("\n=== カスタムアルゴリズムの作成例 ===")
    
    # 実際のモデルから予測データを生成
    print("保存済みモデルから実際の予測データを生成中...")
    y_pred_edges, edges_capacity, commodities = create_sample_data()
    
    custom_algorithm = DemandPriorityBeamSearch(
        y_pred_edges=y_pred_edges,
        beam_size=3,
        batch_size=y_pred_edges.shape[0],
        edges_capacity=edges_capacity,
        commodities=commodities,
        dtypeFloat=torch.float32,
        dtypeLong=torch.long,
        mode_strict=False,
        max_iter=5
    )
    
    commodity_paths, is_feasible = custom_algorithm.search()
    performance_info = custom_algorithm.get_performance_info()
    
    print(f"カスタムアルゴリズム実行時間: {performance_info['execution_time']:.4f}秒")
    print(f"実行可能: {is_feasible}")
    print(f"見つかったパス数: {sum(len(paths) for paths in commodity_paths)}")
    
    # データの詳細情報
    print(f"\nデータ情報:")
    print(f"  バッチサイズ: {y_pred_edges.shape[0]}")
    print(f"  ノード数: {y_pred_edges.shape[1]}")
    print(f"  コモディティ数: {y_pred_edges.shape[2]}")
    print(f"  予測テンソル形状: {y_pred_edges.shape}")

if __name__ == "__main__":
    example_custom_algorithm()