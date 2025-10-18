import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import time

class BeamSearchAlgorithm(ABC):
    """ビームサーチアルゴリズムの抽象基底クラス"""
    
    def __init__(self, y_pred_edges, beam_size, batch_size, edges_capacity, commodities, 
                 dtypeFloat, dtypeLong, mode_strict=False, max_iter=5):
        # 共通の初期化処理
        y = F.log_softmax(y_pred_edges, dim=4)  # B x V x V x C x voc_edges
        y = y[:, :, :, :, 1]  # B x V x V
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number
        
        self.y = y
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.edges_capacity = edges_capacity
        self.commodities = commodities
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        self.mode_strict = mode_strict
        self.max_iter = max_iter
        
        # パフォーマンス計測用
        self.execution_time = 0.0
        self.algorithm_name = self.__class__.__name__

    def search(self) -> Tuple[List[List[List[int]]], bool]:
        """
        メインの検索メソッド - 共通のフレームワーク
        
        Returns:
            Tuple[List[List[List[int]]], bool]: (all_commodity_paths, is_feasible)
        """
        start_time = time.time()
        
        node_orders = []
        all_commodity_paths = []

        for batch in range(self.batch_size):
            batch_node_orders, commodity_paths, is_feasible = self._search_single_batch(batch)
            node_orders.append(batch_node_orders)
            all_commodity_paths.append(commodity_paths)

        self.execution_time = time.time() - start_time
        return all_commodity_paths, is_feasible

    @abstractmethod
    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """
        単一バッチでの検索 - アルゴリズム固有の実装
        
        Args:
            batch: バッチインデックス
            
        Returns:
            Tuple[List[List[int]], List[List[int]], bool]: (node_orders, commodity_paths, is_feasible)
        """
        pass

    def get_performance_info(self) -> dict:
        """
        パフォーマンス情報の取得
        
        Returns:
            dict: パフォーマンス情報
        """
        return {
            'algorithm_name': self.algorithm_name,
            'execution_time': self.execution_time,
            'beam_size': self.beam_size,
            'batch_size': self.batch_size,
            'max_iter': self.max_iter
        }


class StandardBeamSearch(BeamSearchAlgorithm):
    """標準的なビームサーチアルゴリズム（元の実装）"""
    
    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """標準的なビームサーチによる単一バッチ検索"""
        batch_y_pred_edges = self.y[batch]
        commodities = self.commodities[batch]
        count = 0

        while count < self.max_iter:
            batch_edges_capacity = self.edges_capacity[batch]
            
            # ランダムシャッフルによる前処理
            random_indices = torch.randperm(commodities.size(0))
            shuffled_commodities = commodities[random_indices]
            shuffled_pred_edges = batch_y_pred_edges[:, :, random_indices]
            _, original_indices = torch.sort(random_indices)

            node_orders = []
            commodity_paths = []
            is_feasible = True

            for index, commodity in enumerate(shuffled_commodities):
                source_node = commodity[0].item()
                target_node = commodity[1].item()
                demand = commodity[2].item()
                
                # ビームサーチによるパス探索
                node_order, remaining_edges_capacity, best_path = self._beam_search_for_commodity(
                    batch_edges_capacity, shuffled_pred_edges[:, :, index], 
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
        """標準的なビームサーチによるパス探索"""
        beam_queue = [(source, [source], 0, edges_capacity.clone())]
        best_paths = []

        while beam_queue:
            current_scores = [item[2] for item in beam_queue]
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


class DeterministicBeamSearch(BeamSearchAlgorithm):
    """決定論的なビームサーチアルゴリズム（シャッフルなし）"""
    
    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """決定論的なビームサーチによる単一バッチ検索"""
        batch_y_pred_edges = self.y[batch]
        commodities = self.commodities[batch]
        count = 0

        while count < self.max_iter:
            batch_edges_capacity = self.edges_capacity[batch]
            
            # シャッフルなしの決定論的処理
            original_indices = torch.arange(commodities.size(0))
            processed_commodities = commodities
            processed_pred_edges = batch_y_pred_edges

            node_orders = []
            commodity_paths = []
            is_feasible = True

            for index, commodity in enumerate(processed_commodities):
                source_node = commodity[0].item()
                target_node = commodity[1].item()
                demand = commodity[2].item()
                
                # ビームサーチによるパス探索
                node_order, remaining_edges_capacity, best_path = self._beam_search_for_commodity(
                    batch_edges_capacity, processed_pred_edges[:, :, index], 
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
        """決定論的なビームサーチによるパス探索（元の実装と同じ）"""
        beam_queue = [(source, [source], 0, edges_capacity.clone())]
        best_paths = []

        while beam_queue:
            current_scores = [item[2] for item in beam_queue]
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


class GreedyBeamSearch(BeamSearchAlgorithm):
    """貪欲的なビームサーチアルゴリズム（ビームサイズ1）"""
    
    def __init__(self, y_pred_edges, beam_size, batch_size, edges_capacity, commodities, 
                 dtypeFloat, dtypeLong, mode_strict=False, max_iter=5):
        super().__init__(y_pred_edges, 1, batch_size, edges_capacity, commodities, 
                        dtypeFloat, dtypeLong, mode_strict, max_iter)

    def _search_single_batch(self, batch: int) -> Tuple[List[List[int]], List[List[int]], bool]:
        """貪欲的なビームサーチによる単一バッチ検索"""
        batch_y_pred_edges = self.y[batch]
        commodities = self.commodities[batch]
        count = 0

        while count < self.max_iter:
            batch_edges_capacity = self.edges_capacity[batch]
            
            # ランダムシャッフルによる前処理
            random_indices = torch.randperm(commodities.size(0))
            shuffled_commodities = commodities[random_indices]
            shuffled_pred_edges = batch_y_pred_edges[:, :, random_indices]
            _, original_indices = torch.sort(random_indices)

            node_orders = []
            commodity_paths = []
            is_feasible = True

            for index, commodity in enumerate(shuffled_commodities):
                source_node = commodity[0].item()
                target_node = commodity[1].item()
                demand = commodity[2].item()
                
                # 貪欲的なパス探索（ビームサイズ1）
                node_order, remaining_edges_capacity, best_path = self._greedy_search_for_commodity(
                    batch_edges_capacity, shuffled_pred_edges[:, :, index], 
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

    def _greedy_search_for_commodity(self, edges_capacity, y_commodities, source, target, demand):
        """貪欲的なパス探索（ビームサイズ1）"""
        beam_queue = [(source, [source], 0, edges_capacity.clone())]
        best_paths = []

        while beam_queue:
            current_scores = [item[2] for item in beam_queue]
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


class BeamSearchFactory:
    """ビームサーチアルゴリズムのファクトリークラス"""
    
    ALGORITHMS = {
        'standard': StandardBeamSearch,
        'deterministic': DeterministicBeamSearch,
        'greedy': GreedyBeamSearch
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, **kwargs) -> BeamSearchAlgorithm:
        """
        指定されたアルゴリズムのインスタンスを作成
        
        Args:
            algorithm_name: アルゴリズム名 ('standard', 'deterministic', 'greedy')
            **kwargs: アルゴリズムの初期化パラメータ
            
        Returns:
            BeamSearchAlgorithm: アルゴリズムのインスタンス
            
        Raises:
            ValueError: 無効なアルゴリズム名の場合
        """
        if algorithm_name not in cls.ALGORITHMS:
            available = ', '.join(cls.ALGORITHMS.keys())
            raise ValueError(f"無効なアルゴリズム名: {algorithm_name}. 利用可能: {available}")
        
        algorithm_class = cls.ALGORITHMS[algorithm_name]
        return algorithm_class(**kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """利用可能なアルゴリズムのリストを取得"""
        return list(cls.ALGORITHMS.keys())


# 後方互換性のためのエイリアス
class BeamsearchUELB(StandardBeamSearch):
    """後方互換性のためのエイリアス"""
    pass
