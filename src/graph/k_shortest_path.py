#!/usr/bin/env python3
"""
K Shortest Path utilities for graph routing problems
"""

import networkx as nx
from typing import List, Tuple


class KShortestPathFinder:
    """
    K最短経路探索のためのユーティリティクラス
    """
    
    def __init__(self):
        pass
    
    def search_k_shortest_paths(self, G: nx.Graph, commodity_list: List[List[int]], K: int) -> List[List[List[int]]]:
        """
        各品種に対してK最短経路を探索
        
        Args:
            G: NetworkXグラフ
            commodity_list: 品種リスト [[source, target, demand], ...]
            K: 探索する経路数
            
        Returns:
            全品種のK最短経路のリスト [[[path1], [path2], ...], ...]
        """
        allcommodity_ksps = []
        
        for i, commodity in enumerate(commodity_list):
            source, target, demand = commodity[0], commodity[1], commodity[2]
            ksps_list = self._find_k_shortest_paths_single(G, source, target, K)
            allcommodity_ksps.append(ksps_list)
            
        return allcommodity_ksps
    
    def _find_k_shortest_paths_single(self, G: nx.Graph, source: int, target: int, K: int) -> List[List[int]]:
        """
        単一の品種に対してK最短経路を探索 (Yen's Algorithm使用)
        
        Args:
            G: NetworkXグラフ
            source: 始点
            target: 終点
            K: 探索する経路数
            
        Returns:
            K最短経路のリスト [[path1], [path2], ...]
        """
        # ノードの存在チェック - 強制終了
        if source not in G.nodes():
            raise ValueError(f"FATAL ERROR: Source node {source} not in graph (available nodes: {sorted(G.nodes())})")
        
        if target not in G.nodes():
            raise ValueError(f"FATAL ERROR: Target node {target} not in graph (available nodes: {sorted(G.nodes())})")
        
        if source == target:
            raise ValueError(f"FATAL ERROR: Source and target are the same node {source}")
            
        try:
            # Yen's algorithmを使用してK最短経路を探索
            paths_generator = nx.shortest_simple_paths(G, source, target, weight='weight')
            ksps_list = []
            
            for counter, path in enumerate(paths_generator):
                ksps_list.append(path)
                if counter == K - 1:
                    break
                    
            # K本の経路が見つからない場合は、見つかった分だけ返す
            if len(ksps_list) < K:
                print(f"Warning: Only {len(ksps_list)} paths found for source {source} to target {target}")
                
            return ksps_list
            
        except nx.NetworkXNoPath:
            print(f"Warning: No path found from source {source} to target {target}")
            return []
        except Exception as e:
            print(f"Error in K shortest path search: {e}")
            return []
    
    def get_path_length(self, G: nx.Graph, path: List[int]) -> float:
        """
        経路の長さを計算
        
        Args:
            G: NetworkXグラフ
            path: 経路のノードリスト
            
        Returns:
            経路の総重み
        """
        if len(path) < 2:
            return 0.0
            
        total_length = 0.0
        for i in range(len(path) - 1):
            try:
                edge_data = G[path[i]][path[i+1]]
                weight = edge_data.get('weight', 1.0)
                total_length += weight
            except KeyError:
                print(f"Warning: Edge ({path[i]}, {path[i+1]}) not found in graph")
                return float('inf')
                
        return total_length
    
    def validate_path(self, G: nx.Graph, path: List[int]) -> bool:
        """
        経路が有効かどうかを検証
        
        Args:
            G: NetworkXグラフ
            path: 経路のノードリスト
            
        Returns:
            経路が有効かどうか
        """
        if len(path) < 2:
            return False
            
        for i in range(len(path) - 1):
            if not G.has_edge(path[i], path[i+1]):
                return False
                
        return True
    
    def get_path_capacity_bottleneck(self, G: nx.Graph, path: List[int]) -> float:
        """
        経路のボトルネック容量を取得
        
        Args:
            G: NetworkXグラフ
            path: 経路のノードリスト
            
        Returns:
            経路の最小容量
        """
        if len(path) < 2:
            return float('inf')
            
        min_capacity = float('inf')
        for i in range(len(path) - 1):
            try:
                edge_data = G[path[i]][path[i+1]]
                capacity = edge_data.get('capacity', float('inf'))
                min_capacity = min(min_capacity, capacity)
            except KeyError:
                return 0.0
                
        return min_capacity
    
    def filter_paths_by_capacity(self, G: nx.Graph, paths: List[List[int]], min_capacity: float) -> List[List[int]]:
        """
        指定容量以上の経路のみをフィルタリング
        
        Args:
            G: NetworkXグラフ
            paths: 経路のリスト
            min_capacity: 最小容量要件
            
        Returns:
            フィルタリングされた経路のリスト
        """
        filtered_paths = []
        for path in paths:
            if self.get_path_capacity_bottleneck(G, path) >= min_capacity:
                filtered_paths.append(path)
        return filtered_paths