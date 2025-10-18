#!/usr/bin/env python3
"""
Reinforcement Learning Environment for Graph Convolutional Network UELB
Refactored from deepRLmain.py to integrate with existing codebase
"""

import gymnasium as gym
import networkx as nx
import numpy as np
import random
import csv
import copy
from typing import List, Tuple, Optional

from ..graph.k_shortest_path import KShortestPathFinder


class MinMaxLoadKSPsEnv(gym.core.Env):
    """
    強化学習環境クラス - K最短経路を用いた最大負荷率最小化問題
    """
    
    def __init__(self, config: dict):
        """
        環境の初期化
        
        Args:
            config: 設定辞書
        """
        super().__init__()
        
        self.config = config  # 設定を保存
        self.data_idx = 0  # データインデックス
        self.max_train_data = config.get('num_train_data', 100)  # 設定からデータ数を取得
        
        # パラメータの設定
        self.K = config.get('K', 10)
        self.n_action = config.get('n_action', 20)
        self.max_step = config.get('max_step', 20)
        
        # 実際に使用したデータのインデックスを追跡
        self.current_used_data_idx = 0
        
        # 強化学習パラメータ
        self.initial_state = config.get('initial_state', 1)  # 1: 最短経路, 2: ランダム
        self.reward_state = config.get('reward_state', 2)    # 1: 最大負荷率, 2: 差分
        self.count_limit = config.get('count_limit', 100000)
        
        # 観測・行動空間の定義
        obs_low = config.get('obs_low', -20)
        obs_high = config.get('obs_high', 20)
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.observation_space = gym.spaces.Box(
            low=obs_low, 
            high=obs_high, 
            shape=(self.n_action,),
            dtype=np.float32
        )
        
        # 状態変数の初期化
        self.time = 0
        self.candidate_list = []
        
        # K最短経路探索器
        self.ksp_finder = KShortestPathFinder()
        
        # データ関連の初期化
        self.G = None
        self.commodity_list = []
        self.grouping = []
        self.pair_list = []
        self.allcommodity_ksps = []
        
    
    def search_combination(self, allcommodity_ksps: List[List[List[int]]]) -> List[List[List[int]]]:
        """初期状態：最短経路"""
        comb = []
        for z in range(len(allcommodity_ksps)):
            if len(allcommodity_ksps[z]) > 0:
                comb.append(allcommodity_ksps[z][0])
                # 乱数の数を揃えるため
                if len(allcommodity_ksps[z]) > 1:
                    random.choice(allcommodity_ksps[z])
            else:
                # 経路が見つからない場合は強制終了
                raise ValueError(f"FATAL ERROR: No paths found for commodity {z} (source: {self.commodity_list[z][0]}, target: {self.commodity_list[z][1]})")
        return [comb]
    
    def search_random_combination(self, allcommodity_ksps: List[List[List[int]]]) -> List[List[List[int]]]:
        """初期状態：ランダム"""
        comb = []
        for z in range(len(allcommodity_ksps)):
            if len(allcommodity_ksps[z]) > 0:
                comb.append(random.choice(allcommodity_ksps[z]))
            else:
                # 経路が見つからない場合は強制終了
                raise ValueError(f"FATAL ERROR: No paths found for commodity {z} (source: {self.commodity_list[z][0]}, target: {self.commodity_list[z][1]})")
        return [comb]
    
    def get_pair_list(self) -> List[List]:
        """品種と経路のペアを列挙"""
        pair_list = []
        for c in range(self.commodity):
            for p in range(len(self.allcommodity_ksps[c])):
                path = self.allcommodity_ksps[c][p]
                pair_list.append([c, path])
        return pair_list
    
    def exchange_path_action(self, grouping: List[List[int]], action: int) -> List[List[int]]:
        """行動の値に応じて経路交換"""
        new_grouping = grouping.copy()
        c, path, cost = self.candidate_list[action]
        new_grouping[c] = path
        return new_grouping
    
    def exchange_path_pair(self, grouping: List[List[int]], c: int, path: List[int]) -> List[List[int]]:
        """候補を作成するための経路交換"""
        new_grouping = grouping.copy()
        new_grouping[c] = path
        return new_grouping
    
    def zero_one(self, grouping: List[List[int]]) -> List[List[int]]:
        """負荷率計算時の辺の変換0or1"""
        zo_combination = []
        for l in range(self.commodity):
            x_kakai = [0] * len(self.edges_notindex)
            for a in range(len(grouping[l])):
                if a == len(grouping[l]) - 1:
                    break
                edge_set = (grouping[l][a], grouping[l][a + 1])
                try:
                    idx = self.edges_notindex.index(edge_set)
                    x_kakai[idx] = 1
                except ValueError:
                    # 逆方向の辺も考慮
                    edge_set_rev = (grouping[l][a + 1], grouping[l][a])
                    try:
                        idx = self.edges_notindex.index(edge_set_rev)
                        x_kakai[idx] = 1
                    except ValueError:
                        pass  # 辺が存在しない場合
            zo_combination.append(x_kakai)
        return zo_combination
    
    def load_factor(self, grouping: List[List[int]]) -> List[float]:
        """負荷率計算"""
        loads = []
        zo_combination = self.zero_one(grouping)
        for e in range(len(self.edge_list)):
            load = sum(
                (zo_combination[l][e]) * (self.commodity_list[l][2])
                for l in range(self.commodity)
            ) / self.capacity_list[self.edge_list[e][1]]
            loads.append(load)
        return loads
    
    def max_load_factor(self, grouping: List[List[int]]) -> float:
        """最大負荷率計算"""
        loads = self.load_factor(grouping)
        return max(loads) if loads else 0.0
    
    def get_difference(self, grouping: List[List[int]]) -> float:
        """最大負荷率の変化差"""
        new_maxloadfactor = self.max_load_factor(grouping)
        difference = new_maxloadfactor - self.old_maxloadfactor
        return -1 * difference
    
    def get_reward_maxload(self, grouping: List[List[int]]) -> float:
        """報酬定義１：最大負荷率ベース"""
        return -1 * self.max_load_factor(grouping)
    
    def get_reward_difference(self, grouping: List[List[int]], bf_maxloadfactor: float) -> float:
        """報酬定義２：差分ベース"""
        new_maxloadfactor = self.max_load_factor(grouping)
        difference = new_maxloadfactor - bf_maxloadfactor
        return -1 * difference
    
    def get_observation(self) -> List[float]:
        """観測変数の取得"""
        grouping = self.grouping.copy()
        candidate_list = []
        self.old_maxloadfactor = self.max_load_factor(grouping)
        
        for pair in self.pair_list:
            c, path = pair
            if self.reward_state == 1:
                cost = self.get_reward_maxload(self.exchange_path_pair(grouping, c, path))
                if cost >= -1:  # オーバーフローじゃない場合にエントリー
                    candidate_list.append([c, path, cost])
            elif self.reward_state == 2:
                cost = self.get_difference(self.exchange_path_pair(grouping, c, path))
                candidate_list.append([c, path, cost])
        
        # コストの大きい順に並び替えて個数分取り出す
        self.candidate_list = sorted(candidate_list, key=lambda x: -x[2])[:self.n_action]
        mask = [cand[2] for cand in self.candidate_list]
        
        # エントリー数が足りない場合
        if len(mask) < self.n_action:
            i = self.n_action - len(mask)
            for n in range(i):
                mask.append(-100.0)
        
        return mask
    
    def check_is_done(self) -> bool:
        """終了条件の判定"""
        if self.time >= self.max_step:
            return True
        if all(val == -100.0 for val in self.observation):
            return True
        return False
    
    def step(self, action: int) -> Tuple[List[float], float, bool, dict, float]:
        """1ステップの実行"""
        self.time += 1
        observation = self.get_observation()
        oldmaxload = self.max_load_factor(self.grouping)
        
        if all(val == -100.0 for val in observation):
            done = True
            info = {}
            if self.reward_state == 1:
                self.reward = self.get_reward_maxload(self.grouping)
            elif self.reward_state == 2:
                self.reward = self.get_reward_difference(self.grouping, oldmaxload)
            self.observation = self.get_observation()
        else:
            self.grouping = self.exchange_path_action(self.grouping, action).copy()
            if self.reward_state == 1:
                self.reward = self.get_reward_maxload(self.grouping)
            elif self.reward_state == 2:
                self.reward = self.get_reward_difference(self.grouping, oldmaxload)
            self.observation = self.get_observation()
            done = self.check_is_done()
            info = {}
        
        maxload = self.max_load_factor(self.grouping)
        return self.observation, self.reward, done, info, maxload
    
    def reset(self, mode: str = 'train') -> List[float]:
        """エピソードの初期化"""
        self.time = 0
        # 毎エピソードで新しいグラフ・品種を生成
        self.grouping = self.get_random_grouping(mode)
        self.pair_list = self.get_pair_list()
        self.observation = self.get_observation()
        count = 0
        
        # 初期状態から全ての行動のエントリーが一つも存在しない場合
        while all(val == -100.0 for val in self.observation):
            combination = self.search_random_combination(self.allcommodity_ksps)
            self.grouping = combination[0]
            count += 1
            self.observation = self.get_observation()
            if count == self.count_limit:
                break
        
        return self.observation
    
    def get_random_grouping(self, mode: str = 'train') -> List[List[int]]:
        """状態の初期化 - 既存データを使用"""
        # 現在のdata_idxを保存（このデータを実際に使用する）
        self.current_used_data_idx = self.data_idx
        
        # 既存データを使用（GCNモードと同じデータ範囲）
        self.load_data_from_files(self.data_idx, mode)
        
        # データインデックスの循環処理
        if mode == 'train':
            # トレーニング時：設定のnum_train_dataを使用して循環
            self.data_idx = (self.data_idx + 1) % self.max_train_data
        elif mode == 'test':
            # テスト時：設定のnum_test_dataを使用して循環
            max_test_data = self.config.get('num_test_data', 20)
            self.data_idx = (self.data_idx + 1) % max_test_data
        else:
            # その他（val等）：デフォルトでtrainと同じ
            self.data_idx = (self.data_idx + 1) % self.max_train_data
        
        # K最短経路探索
        self.allcommodity_ksps = self.ksp_finder.search_k_shortest_paths(
            self.G, self.commodity_list, self.K
        )
        
        # 初期状態の設定
        if self.initial_state == 1:
            combination = self.search_combination(self.allcommodity_ksps)
        elif self.initial_state == 2:
            combination = self.search_random_combination(self.allcommodity_ksps)
        else:
            combination = self.search_combination(self.allcommodity_ksps)
        
        self.grouping = combination[0]
        return self.grouping
    
    def load_data_from_files(self, data_idx: int, mode: str = 'train') -> None:
        """
        dataディレクトリから既存のデータを読み込み
        
        Args:
            data_idx: データインデックス
            mode: データモード ('train', 'test', 'val')
        """
        # グラフファイルの読み込み（GCNモードと同じ形式）
        graph_file = f"./data/{mode}_data/graph_file/{data_idx-(data_idx%10)}/graph_{data_idx}.gml"
        self.G = nx.read_gml(graph_file, destringizer=int)
        
        # 品種ファイルの読み込み（GCNモードと同じ形式）
        commodity_file = f"./data/{mode}_data/commodity_file/{data_idx-(data_idx%10)}/commodity_data_{data_idx}.csv"
        
        # デバッグ出力: RLがどのファイルを読み込んでいるか
        # print(f"    RL loading: graph={graph_file}, commodity={commodity_file}")
        self.commodity_list = []
        available_nodes = set(self.G.nodes())
        
        with open(commodity_file, 'r') as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if len(row) >= 3:
                    source, target, demand = int(row[0]), int(row[1]), int(row[2])
                    
                    # ノードID不一致の検出と強制終了
                    if source not in available_nodes:
                        raise ValueError(f"FATAL ERROR: Source node {source} in commodity data (file: {commodity_file}, row: {row_idx}) does not exist in graph (available nodes: {sorted(available_nodes)})")
                    
                    if target not in available_nodes:
                        raise ValueError(f"FATAL ERROR: Target node {target} in commodity data (file: {commodity_file}, row: {row_idx}) does not exist in graph (available nodes: {sorted(available_nodes)})")
                    
                    if source == target:
                        raise ValueError(f"FATAL ERROR: Source and target are the same node {source} in commodity data (file: {commodity_file}, row: {row_idx})")
                    
                    self.commodity_list.append([source, target, demand])
        
        # 品種データが空の場合もエラー
        if len(self.commodity_list) == 0:
            raise ValueError(f"FATAL ERROR: No valid commodities found in file {commodity_file}")
        
        self.commodity = len(self.commodity_list)
        self.node = len(self.G.nodes())
        
        # グラフの属性取得
        self.capacity_list = nx.get_edge_attributes(self.G, 'capacity')
        self.edge_list = list(enumerate(self.G.edges()))
        self.edges_notindex = [self.edge_list[z][1] for z in range(len(self.edge_list))]
    
    
    def render(self, mode='human'):
        """画面への描画・可視化"""
        pass
    
    def close(self):
        """終了時の処理"""
        pass
    
    def seed(self, seed=None):
        """乱数の固定"""
        random.seed(seed)
        np.random.seed(seed)
        return [seed]