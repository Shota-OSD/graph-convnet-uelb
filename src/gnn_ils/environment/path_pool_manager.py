import networkx as nx
import torch
from typing import List

from src.common.graph.k_shortest_path import KShortestPathFinder


class PathPoolManager:
    """
    候補パスプールの構築と管理。

    各コモディティに対して KSP (Yen's Algorithm) + link-disjoint paths を統合し、
    重複排除した候補パスプールを構築する。
    """

    def __init__(self, config: dict):
        """
        Args:
            config:
                - K: KSP の K (default: 10)
                - max_disjoint: link-disjoint の最大数 (default: 5)
                - max_candidate_paths: コモディティあたり最大候補数 (default: 15)
        """
        self.K = config.get('K', 10)
        self.max_disjoint = config.get('max_disjoint', 5)
        self.max_candidate_paths = config.get('max_candidate_paths', 15)
        self.ksp_finder = KShortestPathFinder()

    def build_path_pool(
        self,
        G: nx.Graph,
        commodity_list: List[List[int]],
    ) -> List[List[List[int]]]:
        """
        全コモディティの候補パスプールを構築する。

        Args:
            G: NetworkXグラフ
            commodity_list: [[src, dst, demand], ...]

        Returns:
            path_pool: [C][P_c][path_length]
        """
        path_pool = []
        for commodity in commodity_list:
            src, dst = int(commodity[0]), int(commodity[1])

            ksp_paths = self.ksp_finder._find_k_shortest_paths_single(G, src, dst, self.K)
            disjoint_paths = self._find_link_disjoint_paths(G, src, dst, self.max_disjoint)
            merged = self._merge_and_deduplicate(ksp_paths, disjoint_paths, self.max_candidate_paths)

            if not merged:
                try:
                    fallback = nx.dijkstra_path(G, src, dst, weight='weight')
                    merged = [list(fallback)]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    merged = [[src]]

            path_pool.append(merged)

        return path_pool

    def _find_link_disjoint_paths(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        max_paths: int,
    ) -> List[List[int]]:
        """
        Link-disjoint paths を探索する。

        nx.edge_disjoint_paths を使用。失敗時は空リストを返す。
        """
        try:
            paths_gen = nx.edge_disjoint_paths(G, source, target)
            disjoint_paths = []
            for path in paths_gen:
                disjoint_paths.append(list(path))
                if len(disjoint_paths) >= max_paths:
                    break
            return disjoint_paths
        except Exception:
            return []

    def _merge_and_deduplicate(
        self,
        ksp_paths: List[List[int]],
        disjoint_paths: List[List[int]],
        max_total: int,
    ) -> List[List[int]]:
        """
        KSP と disjoint paths を統合し、重複を排除する。

        優先順位: KSP (短い順) → disjoint paths
        """
        seen = set()
        merged = []

        for path in ksp_paths + disjoint_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen and len(path) >= 2:
                seen.add(path_tuple)
                merged.append(path)
                if len(merged) >= max_total:
                    break

        return merged

    def get_path_mask(
        self,
        path_pool: List[List[List[int]]],
        max_paths: int,
    ) -> torch.Tensor:
        """
        有効パスのマスクテンソルを生成する。

        Returns:
            path_mask: [C, max_paths] - bool
        """
        num_commodities = len(path_pool)
        mask = torch.zeros(num_commodities, max_paths, dtype=torch.bool)
        for c, paths in enumerate(path_pool):
            num_valid = min(len(paths), max_paths)
            mask[c, :num_valid] = True
        return mask
