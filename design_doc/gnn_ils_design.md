# GNN-ILS 詳細実装設計書

## 1. 概要

### 1.1 コンセプト

GNN-ILS (GNN-guided Iterated Local Search) は、GCN・RL-KSP・SeqFlowRL に続く4つ目のアプローチとして、**全コモディティ到達率100%を構造的に保証**しながら最大負荷率を最小化する手法である。

核心となるアイデア:
- **初期解**: KSP (K Shortest Paths) + Link-disjoint paths でコモディティごとに候補パスプールを構築し、最短パスで初期割当を決定する。この時点で全コモディティの到達が保証される。
- **改善ループ (ILS)**: GNN が学習したポリシーに従い、2段階の意思決定 — (1) どのコモディティのパスを交換するか、(2) どの候補パスに変更するか — を繰り返して max load factor を削減する。
- **到達率100%保証**: パスプール内の有効パスしか選択できないため、改善ループ中に到達が壊れることは構造的にない。

### 1.2 既存手法との位置づけ

| 手法 | アプローチ | 到達率保証 | 解の改善手段 |
|---|---|---|---|
| GCN | エッジスコア予測 + ビームサーチ | なし | ビーム幅拡大 |
| RL-KSP | DQN でパス交換 | 100% (KSP依存) | epsilon-greedy 探索 |
| SeqFlowRL | GNN + A2C 逐次ルーティング | なし | エントロピーボーナス |
| **GNN-ILS** | GNN + 2段階A2C + ILS | **100% (構造保証)** | ILS改善ループ |

### 1.3 アーキテクチャ概観

```
PathPoolManager
  ├── KShortestPathFinder (Yen's Algorithm, 既存再利用)
  └── LinkDisjointPathFinder (nx.edge_disjoint_paths, 新規)
        ↓
  候補パスプール [C][P_c] (コモディティ毎に可変数のパス)
        ↓
ILSEnvironment (初期解 = 最短パス割当)
        ↓
  ┌─── 改善ループ (max_iterations 回) ───┐
  │                                       │
  │  GNNILSEncoder (グラフ状態エンコード)  │
  │    ├── node_features  [1, V, C, H]    │
  │    ├── edge_features  [1, V, V, C, H] │
  │    └── graph_embedding [1, H]          │
  │        ↓                               │
  │  Level1: CommoditySelector             │
  │    → どのコモディティのパスを交換? [C] │
  │        ↓                               │
  │  Level2: PathSelector                  │
  │    → どの候補パスに変更? [P_c]         │
  │        ↓                               │
  │  ValueHead → V(s)                      │
  │        ↓                               │
  │  パス交換 → 負荷再計算                 │
  │        ↓                               │
  │  改善なし → 打ち切り or 次イテレーション│
  └────────────────────────────────────────┘
        ↓
  最終解 (load_factor, paths)
```

---

## 2. ディレクトリ構成

```
src/gnn_ils/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── gnn_ils_encoder.py       # GNNILSEncoder: グラフエンコーダ
│   ├── commodity_selector.py    # CommoditySelector: Level1 ポリシー
│   ├── path_selector.py         # PathSelector: Level2 ポリシー
│   ├── value_head.py            # ILSValueHead: Critic
│   └── gnn_ils_model.py         # GNNILSModel: 統合モデル
├── environment/
│   ├── __init__.py
│   ├── ils_environment.py       # ILSEnvironment: ILS改善ループ環境
│   └── path_pool_manager.py     # PathPoolManager: 候補パス管理
├── training/
│   ├── __init__.py
│   ├── ils_a2c_strategy.py      # ILSA2CStrategy: 2段階A2C損失
│   └── trainer.py               # GNNILSTrainer: 学習ループ
└── utils/
    ├── __init__.py
    └── load_utils.py            # 負荷計算ユーティリティ

scripts/gnn_ils/
├── train_gnn_ils.py             # 学習エントリーポイント
└── test_gnn_ils.py              # テストエントリーポイント

configs/gnn_ils/
└── gnn_ils_base.json            # 基本設定ファイル
```

---

## 3. モデルアーキテクチャ

### 3.1 GNNILSEncoder

グラフの静的情報 (容量) と動的情報 (現在の負荷) をエンコードする。
`HybridGNNEncoder` と同一構造だが、ILS固有の入力特徴量 (現在のパス割当に基づく負荷分布) に対応するため新規作成する。
`ResidualGatedGCNLayer` は直接再利用する。

```python
# src/gnn_ils/models/gnn_ils_encoder.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from src.gcn.models.gcn_layers import ResidualGatedGCNLayer


class GNNILSEncoder(nn.Module):
    """
    GNN Encoder for GNN-ILS.

    HybridGNNEncoder と同一構造だが、ILS 固有の入力に対応:
    - edge_usage は ILS ループ中に毎ステップ更新される
    - per_commodity_load: コモディティ毎のエッジ使用量 [B, V, V, C]
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 設定辞書
                - num_nodes: int
                - num_commodities: int
                - hidden_dim: int (default: 128)
                - num_layers: int (default: 8)
                - aggregation: str (default: 'mean')
                - dropout_rate: float (default: 0.3)
        """
        super().__init__()
        ...

    def forward(
        self,
        x_nodes: Tensor,             # [B, V, C] - torch.long
        x_commodities: Tensor,       # [B, C, 3] - torch.float (src, dst, demand)
        x_edges_capacity: Tensor,    # [B, V, V] - torch.float
        x_edges_usage: Tensor,       # [B, V, V] - torch.float (ILS状態)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            node_features:   [B, V, C, H] - torch.float
            edge_features:   [B, V, V, C, H] - torch.float
            graph_embedding: [B, H] - torch.float
        """
        ...
```

**内部構成** (HybridGNNEncoder と同一):
- `nodes_embedding`: `nn.Embedding(num_commodities * 3, hidden_dim // 2)`
- `commodities_embedding`: `nn.Linear(1, hidden_dim // 2, bias=False)`
- `edge_capacity_embedding`: `nn.Linear(1, hidden_dim // 2, bias=False)`
- `edge_usage_embedding`: `nn.Linear(1, hidden_dim // 2, bias=False)`
- `gcn_layers`: `nn.ModuleList([ResidualGatedGCNLayer(hidden_dim, aggregation)] * num_layers)`
- `dropout`: `nn.Dropout(dropout_rate)`

### 3.2 CommoditySelector (Level1 Policy)

どのコモディティのパスを交換するかを選択する Actor。

```python
# src/gnn_ils/models/commodity_selector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.gcn.models.gcn_layers import MLP


class CommoditySelector(nn.Module):
    """
    Level1 Policy: コモディティ選択。

    graph_embedding + per-commodity 情報から、交換対象のコモディティを選択する。
    """

    def __init__(self, hidden_dim: int, num_commodities: int, mlp_layers: int = 2):
        """
        Args:
            hidden_dim: GNNエンコーダの隠れ次元
            num_commodities: コモディティ数
            mlp_layers: MLP層数 (default: 2)
        """
        super().__init__()
        ...

    def forward(
        self,
        node_features: Tensor,      # [B, V, C, H]
        graph_embedding: Tensor,    # [B, H]
        commodity_mask: Tensor,     # [B, C] - bool (交換可能なコモディティ)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            node_features: ノード埋め込み [B, V, C, H]
            graph_embedding: グラフ埋め込み [B, H]
            commodity_mask: 交換可能コモディティのマスク [B, C]
                           (候補パスが1本以下のコモディティはFalse)

        Returns:
            action_probs: コモディティ選択確率 [B, C]
            log_probs: 対数確率 [B, C]
            entropy: エントロピー [B]
        """
        ...

    def _compute_commodity_features(
        self,
        node_features: Tensor,      # [B, V, C, H]
        graph_embedding: Tensor,    # [B, H]
    ) -> Tensor:
        """
        各コモディティの集約特徴量を計算。

        Returns:
            commodity_features: [B, C, 2H]
                (commodity_node_feat [B, C, H] + graph_embedding [B, H] を concat)
        """
        ...
```

**内部構成**:
- `commodity_mlp`: `MLP(hidden_dim * 2, 1, num_layers=mlp_layers, hidden_dims=[128])` — 各コモディティのスコアを算出
- 入力: コモディティ毎のノード集約特徴 `[B, C, H]` と `graph_embedding [B, H]` を concat → `[B, C, 2H]`
- 出力: softmax → 確率分布 `[B, C]`

### 3.3 PathSelector (Level2 Policy)

選択されたコモディティに対して、どの候補パスに変更するかを選択する Actor。

```python
# src/gnn_ils/models/path_selector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from src.gcn.models.gcn_layers import MLP


class PathSelector(nn.Module):
    """
    Level2 Policy: パス選択。

    選択されたコモディティに対し、候補パスプールから最適なパスを選択する。
    パスは可変長のため、パス特徴量をエッジ特徴量の集約で表現する。
    """

    def __init__(self, hidden_dim: int, max_paths: int, mlp_layers: int = 2):
        """
        Args:
            hidden_dim: GNNエンコーダの隠れ次元
            max_paths: コモディティあたりの最大候補パス数 (K + disjoint)
            mlp_layers: MLP層数 (default: 2)
        """
        super().__init__()
        ...

    def forward(
        self,
        edge_features: Tensor,             # [B, V, V, C, H]
        graph_embedding: Tensor,           # [B, H]
        selected_commodity: Tensor,        # [B] - int (Level1で選択されたコモディティ)
        candidate_paths: List[List[List[int]]],  # [B][P_c][path_length]
        path_mask: Tensor,                 # [B, max_paths] - bool
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            edge_features: エッジ埋め込み [B, V, V, C, H]
            graph_embedding: グラフ埋め込み [B, H]
            selected_commodity: 選択済みコモディティ [B]
            candidate_paths: 候補パスのリスト [B][P_c][path_length]
            path_mask: 有効パスマスク [B, max_paths]

        Returns:
            action_probs: パス選択確率 [B, max_paths]
            log_probs: 対数確率 [B, max_paths]
            entropy: エントロピー [B]
        """
        ...

    def _encode_path(
        self,
        edge_features: Tensor,      # [B, V, V, C, H]
        path: List[int],            # [path_length]
        commodity_idx: int,
        batch_idx: int,
    ) -> Tensor:
        """
        パスをエッジ特徴量の集約で表現する。

        path 上の各エッジ (u, v) の edge_features を mean-pooling して
        パス全体の特徴量を得る。

        Returns:
            path_feature: [H]
        """
        ...
```

**内部構成**:
- `path_score_mlp`: `MLP(hidden_dim * 2, 1, num_layers=mlp_layers, hidden_dims=[128])` — パススコアを算出
- 入力: パス特徴量 `[H]` + graph_embedding `[H]` → concat `[2H]`
- パス特徴量: `edge_features[b, u, v, c, :]` を path 上の全エッジについて mean-pooling → `[H]`
- 出力: masked softmax → 確率分布 `[B, max_paths]`

### 3.4 ILSValueHead (Critic)

ILS 改善ループの現在状態に対する価値関数。
既存 `ValueHead` と同一構造で、`graph_embedding` ベースの軽量版も提供。

```python
# src/gnn_ils/models/value_head.py

import torch
import torch.nn as nn
from torch import Tensor

from src.gcn.models.gcn_layers import MLP


class ILSValueHead(nn.Module):
    """
    Critic for GNN-ILS.

    グラフ全体の状態から単一スカラー V(s) を予測する。
    """

    def __init__(self, hidden_dim: int, num_nodes: int, num_commodities: int,
                 mlp_layers: int = 3, use_graph_embedding: bool = False):
        """
        Args:
            hidden_dim: 隠れ次元
            num_nodes: ノード数
            num_commodities: コモディティ数
            mlp_layers: MLP層数 (default: 3)
            use_graph_embedding: True の場合 graph_embedding [B, H] を入力に使う
                                 False の場合 node_features を flatten して使う
        """
        super().__init__()
        ...

    def forward(
        self,
        node_features: Tensor,          # [B, V, C, H]
        graph_embedding: Tensor = None,  # [B, H] (use_graph_embedding=True の場合)
    ) -> Tensor:
        """
        Returns:
            state_value: [B]
        """
        ...
```

**内部構成** (use_graph_embedding=False, デフォルト):
- `value_mlp`: `MLP(V * C * H, 1, num_layers=3, hidden_dims=[512, 256])`

**内部構成** (use_graph_embedding=True):
- `value_mlp`: `MLP(H, 1, num_layers=3, hidden_dims=[256, 128])`

### 3.5 GNNILSModel (統合モデル)

全サブモジュールを統合した Actor-Critic モデル。

```python
# src/gnn_ils/models/gnn_ils_model.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional

from .gnn_ils_encoder import GNNILSEncoder
from .commodity_selector import CommoditySelector
from .path_selector import PathSelector
from .value_head import ILSValueHead


class GNNILSModel(nn.Module):
    """
    GNN-ILS Actor-Critic Model.

    Architecture:
        GNNILSEncoder (shared)
            ├── CommoditySelector (Level1 Actor)
            ├── PathSelector (Level2 Actor)
            └── ILSValueHead (Critic)
    """

    def __init__(self, config: dict, dtypeFloat=torch.float32, dtypeLong=torch.long):
        """
        Args:
            config: 設定辞書
                - num_nodes: int
                - num_commodities: int
                - hidden_dim: int (default: 128)
                - num_layers: int (default: 8)
                - aggregation: str (default: 'mean')
                - dropout_rate: float (default: 0.3)
                - max_candidate_paths: int (default: 15)
                - commodity_selector_mlp_layers: int (default: 2)
                - path_selector_mlp_layers: int (default: 2)
                - value_head_mlp_layers: int (default: 3)
                - use_graph_embedding_value: bool (default: False)
        """
        super().__init__()
        ...

    def encode(
        self,
        x_nodes: Tensor,             # [B, V, C] - torch.long
        x_commodities: Tensor,       # [B, C, 3] - torch.float
        x_edges_capacity: Tensor,    # [B, V, V] - torch.float
        x_edges_usage: Tensor,       # [B, V, V] - torch.float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        GNNエンコード。

        Returns:
            node_features:   [B, V, C, H]
            edge_features:   [B, V, V, C, H]
            graph_embedding: [B, H]
        """
        ...

    def select_commodity(
        self,
        node_features: Tensor,      # [B, V, C, H]
        graph_embedding: Tensor,    # [B, H]
        commodity_mask: Tensor,     # [B, C]
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Level1: コモディティ選択。

        Returns:
            selected_commodity: [B] - int
            log_prob: [B]
            entropy: [B]
        """
        ...

    def select_path(
        self,
        edge_features: Tensor,                    # [B, V, V, C, H]
        graph_embedding: Tensor,                   # [B, H]
        selected_commodity: Tensor,                # [B]
        candidate_paths: List[List[List[int]]],    # [B][P_c][path_length]
        path_mask: Tensor,                         # [B, max_paths]
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Level2: パス選択。

        Returns:
            selected_path_idx: [B] - int
            log_prob: [B]
            entropy: [B]
        """
        ...

    def get_value(
        self,
        node_features: Tensor,          # [B, V, C, H]
        graph_embedding: Tensor = None, # [B, H]
    ) -> Tensor:
        """
        Critic: 状態価値。

        Returns:
            state_value: [B]
        """
        ...
```

---

## 4. 環境設計

### 4.1 PathPoolManager

コモディティごとの候補パスプールを構築・管理する。

```python
# src/gnn_ils/environment/path_pool_manager.py

import networkx as nx
from typing import List, Dict, Tuple

from src.common.graph.k_shortest_path import KShortestPathFinder


class PathPoolManager:
    """
    候補パスプールの構築と管理。

    各コモディティに対して:
    1. K最短パス (Yen's Algorithm, 既存 KShortestPathFinder を再利用)
    2. Link-disjoint paths (nx.edge_disjoint_paths で新規実装)
    を統合し、重複排除した候補パスプールを構築する。
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 設定辞書
                - K: int (KSP の K, default: 10)
                - max_disjoint: int (link-disjoint の最大数, default: 5)
                - max_candidate_paths: int (コモディティあたり最大候補数, default: 15)
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
        全コモディティの候補パスプールを構築。

        Args:
            G: NetworkXグラフ
            commodity_list: [[src, dst, demand], ...]

        Returns:
            path_pool: [C][P_c][path_length]
                       コモディティ毎に可変数の候補パス
        """
        ...

    def _find_link_disjoint_paths(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        max_paths: int,
    ) -> List[List[int]]:
        """
        Link-disjoint paths を探索。

        nx.edge_disjoint_paths を使用。
        disjoint 計算が失敗した場合は空リストを返す。

        Args:
            G: NetworkXグラフ
            source: 始点
            target: 終点
            max_paths: 最大パス数

        Returns:
            disjoint_paths: [[path1], [path2], ...]
        """
        ...

    def _merge_and_deduplicate(
        self,
        ksp_paths: List[List[int]],
        disjoint_paths: List[List[int]],
        max_total: int,
    ) -> List[List[int]]:
        """
        KSP と disjoint paths を統合し、重複を排除。

        優先順位: KSP (短い順) → disjoint paths

        Returns:
            merged_paths: 重複排除済みの候補パスリスト
        """
        ...

    def get_path_mask(
        self,
        path_pool: List[List[List[int]]],
        max_paths: int,
    ) -> 'Tensor':
        """
        有効パスのマスクテンソルを生成。

        Returns:
            path_mask: [C, max_paths] - bool
        """
        ...
```

### 4.2 ILSEnvironment

ILS 改善ループの環境。Gymnasium は使用せず独自ループで実装する。
理由: テンソルベースの状態空間が `Box`/`Discrete` に収まらないため。

```python
# src/gnn_ils/environment/ils_environment.py

import torch
import networkx as nx
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple, Optional

from .path_pool_manager import PathPoolManager
from ..utils.load_utils import compute_load_factor, compute_edge_usage


class ILSEnvironment:
    """
    ILS 改善ループ環境。

    batch_size=1 で動作: ILS は1サンプルずつ改善ループを回す。

    状態空間:
        - x_nodes: [1, V, C] - torch.long
        - x_commodities: [1, C, 3] - torch.float
        - x_edges_capacity: [1, V, V] - torch.float
        - x_edges_usage: [1, V, V] - torch.float (動的に更新)
        - current_assignment: List[List[int]] - 現在のパス割当 [C][path_length]

    行動空間:
        - Level1: コモディティ選択 Discrete(C)
        - Level2: パス選択 Discrete(max_candidate_paths)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 設定辞書
                - num_nodes: int
                - num_commodities: int
                - max_iterations: int (ILS改善ステップ数, default: 50)
                - no_improve_patience: int (改善なし打ち切り, default: 10)
                - perturbation_prob: float (摂動確率, default: 0.1)
        """
        ...

    def reset(
        self,
        G: nx.Graph,
        commodity_list: List[List[int]],
        x_nodes: Tensor,             # [1, V, C] - torch.long
        x_commodities: Tensor,       # [1, C, 3] - torch.float
        x_edges_capacity: Tensor,    # [1, V, V] - torch.float
    ) -> Dict[str, any]:
        """
        エピソードの初期化。

        1. PathPoolManager で候補パスプール構築
        2. 最短パスで初期割当を決定
        3. 初期負荷を計算

        Returns:
            state: {
                'x_nodes': Tensor [1, V, C],
                'x_commodities': Tensor [1, C, 3],
                'x_edges_capacity': Tensor [1, V, V],
                'x_edges_usage': Tensor [1, V, V],
                'current_assignment': List[List[int]],
                'path_pool': List[List[List[int]]],
                'commodity_mask': Tensor [1, C],
                'load_factor': float,
            }
        """
        ...

    def step(
        self,
        selected_commodity: int,
        selected_path_idx: int,
    ) -> Tuple[Dict, float, bool, Dict]:
        """
        1ステップ実行: パス交換 → 負荷再計算。

        Args:
            selected_commodity: 交換対象コモディティ
            selected_path_idx: 新パスのインデックス (path_pool 内)

        Returns:
            state: 更新後の状態辞書
            reward: float (改善ベース報酬)
            done: bool (終了条件)
            info: Dict (メトリクス)
        """
        ...

    def _compute_reward(
        self,
        old_load_factor: float,
        new_load_factor: float,
    ) -> float:
        """
        報酬計算。

        shared モード (デフォルト):
            reward = -(new_load_factor - old_load_factor)
            改善時は正、悪化時は負

        decomposed モード:
            Level1 reward = -(new_load_factor - old_load_factor)
            Level2 reward = -(new_load_factor - old_load_factor)
            (同一だが、将来の拡張のためにメソッドを分離)

        Returns:
            reward: float
        """
        ...

    def _check_done(self) -> bool:
        """
        終了条件の判定。

        - max_iterations に到達
        - no_improve_patience 回連続で改善なし

        Returns:
            done: bool
        """
        ...

    def get_commodity_mask(self) -> Tensor:
        """
        交換可能なコモディティのマスクを生成。

        候補パスが1本以下のコモディティは交換不可。
        現在のパスと異なるパスが存在しないコモディティも交換不可。

        Returns:
            mask: [1, C] - bool
        """
        ...

    def get_path_mask(self, commodity_idx: int) -> Tensor:
        """
        指定コモディティの有効パスマスクを生成。

        Returns:
            mask: [1, max_candidate_paths] - bool
        """
        ...
```

### 4.3 報酬設計

| モード | Level1 報酬 | Level2 報酬 | 備考 |
|---|---|---|---|
| **shared** (デフォルト) | `-(new_lf - old_lf)` | 同一 | 両レベルで同一の改善ベース報酬 |
| **decomposed** | `-(new_lf - old_lf)` | `-(new_lf - old_lf)` | 将来的にレベル別報酬に分離可能 |

- 改善時: `reward > 0`
- 悪化時: `reward < 0`
- 変化なし: `reward = 0`

---

## 5. 学習戦略

### 5.1 ILSA2CStrategy

2段階 A2C 損失計算。Level1 (コモディティ選択) と Level2 (パス選択) の損失を統合する。

```python
# src/gnn_ils/training/ils_a2c_strategy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

from ..models.gnn_ils_model import GNNILSModel
from ..environment.ils_environment import ILSEnvironment


class ILSA2CStrategy:
    """
    2段階 A2C Training Strategy for GNN-ILS.

    ILS改善ループ内で:
    1. Level1 (CommoditySelector) + Level2 (PathSelector) のログ確率を蓄積
    2. ValueHead の予測値を蓄積
    3. エピソード終了時に A2C 損失を計算

    損失 = L_actor_l1 + L_actor_l2 + value_loss_weight * L_critic
           + entropy_weight * (L_entropy_l1 + L_entropy_l2)
    """

    def __init__(self, model: GNNILSModel, config: dict):
        """
        Args:
            model: GNNILSModel instance
            config: 設定辞書
                - learning_rate: float (default: 0.0005)
                - entropy_weight: float (default: 0.01)
                - value_loss_weight: float (default: 0.5)
                - gamma: float (default: 0.99)
                - normalize_advantages: bool (default: True)
                - grad_clip_norm: float (default: 1.0)
                - reward_mode: str ('shared' or 'decomposed', default: 'shared')
        """
        ...

    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        1サンプルの ILS エピソードを実行し、損失を計算・更新。

        batch_size=1 で動作。外側のループで複数サンプルを処理する。

        Args:
            batch_data: {
                'x_nodes': Tensor [1, V, C],
                'x_commodities': Tensor [1, C, 3],
                'x_edges_capacity': Tensor [1, V, V],
                'graph': nx.Graph,
                'commodity_list': List[List[int]],
            }

        Returns:
            metrics: {
                'total_loss': float,
                'actor_l1_loss': float,
                'actor_l2_loss': float,
                'critic_loss': float,
                'entropy_l1': float,
                'entropy_l2': float,
                'mean_reward': float,
                'final_load_factor': float,
                'improvement': float,
                'num_iterations': int,
            }
        """
        ...

    def _run_ils_episode(
        self,
        batch_data: Dict,
    ) -> Dict:
        """
        ILS エピソードを実行し、trajectoryを収集。

        Returns:
            trajectory: {
                'log_probs_l1': List[Tensor],   # [T] 各ステップの Level1 log_prob
                'log_probs_l2': List[Tensor],   # [T] 各ステップの Level2 log_prob
                'entropies_l1': List[Tensor],   # [T]
                'entropies_l2': List[Tensor],   # [T]
                'state_values': List[Tensor],   # [T]
                'rewards': List[float],         # [T]
                'final_load_factor': float,
                'initial_load_factor': float,
            }
        """
        ...

    def _compute_a2c_loss(self, trajectory: Dict) -> Tuple[Tensor, Dict]:
        """
        A2C 損失計算。

        1. discounted returns R_t = sum_{k=0}^{T-t} gamma^k * r_{t+k} を計算
        2. advantage A_t = R_t - V(s_t)
        3. Actor loss (L1 + L2) = -E[log_prob * advantage]
        4. Critic loss = MSE(V(s_t), R_t)
        5. Entropy bonus = -E[entropy]

        Returns:
            total_loss: Tensor (スカラー)
            loss_components: Dict
        """
        ...

    def eval_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        評価ステップ (勾配なし、deterministic)。

        Returns:
            metrics: {
                'final_load_factor': float,
                'improvement': float,
                'num_iterations': int,
                'complete_rate': float,
                'approximation_ratio': float or None,
            }
        """
        ...

    def step_scheduler(self) -> None:
        """学習率スケジューラをステップ。"""
        ...

    def get_current_lr(self) -> float:
        """現在の学習率を取得。"""
        ...
```

### 5.2 損失関数の詳細

```
Total Loss = L_actor_l1 + L_actor_l2 + 0.5 * L_critic - 0.01 * (H_l1 + H_l2)

L_actor_l1 = -mean(log_prob_l1 * advantage)    # Level1: コモディティ選択
L_actor_l2 = -mean(log_prob_l2 * advantage)    # Level2: パス選択
L_critic   = MSE(V(s_t), R_t)                  # Critic: 状態価値
H_l1       = mean(entropy_l1)                   # Level1: エントロピーボーナス
H_l2       = mean(entropy_l2)                   # Level2: エントロピーボーナス

advantage  = R_t - V(s_t).detach()              # V(s_t) は baseline として使用
R_t        = sum_{k=0}^{T-t} gamma^k * r_{t+k}  # discounted return
```

---

## 6. トレーナー

### 6.1 GNNILSTrainer

```python
# src/gnn_ils/training/trainer.py

import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ..models.gnn_ils_model import GNNILSModel
from .ils_a2c_strategy import ILSA2CStrategy
from src.common.types import BatchData, validate_batch_types
from src.common.config.paths import get_model_root


class GNNILSTrainer:
    """
    GNN-ILS Trainer.

    SeqFlowRLTrainer と同様の構成だが、以下の点が異なる:
    - batch_size=1 でサンプル単位にILSエピソードを実行
    - ILS改善ループの統計情報を追加で記録
    """

    def __init__(self, config: dict, dtypeFloat=torch.float32, dtypeLong=torch.long):
        """
        Args:
            config: 設定辞書
            dtypeFloat: float型
            dtypeLong: long型
        """
        ...

    def _setup_device(self) -> torch.device:
        """計算デバイスのセットアップ。"""
        ...

    def _instantiate_model(self) -> GNNILSModel:
        """モデルのインスタンス化。"""
        ...

    def train(self, train_loader, val_loader=None, num_epochs=None) -> Dict:
        """
        メイン学習ループ。

        各エポック:
        1. train_loader から1サンプルずつ取得
        2. ILSEnvironment.reset() で初期解生成
        3. ILS改善ループ実行 (ILSA2CStrategy.train_step)
        4. メトリクス蓄積
        5. val_loader で評価

        Args:
            train_loader: DatasetReader (batch_size=1)
            val_loader: DatasetReader (batch_size=1)
            num_epochs: エポック数

        Returns:
            training_history: Dict
        """
        ...

    def _prepare_batch(self, batch: Any) -> Dict:
        """
        DotDict を ILS 用の辞書に変換。

        SeqFlowRLTrainer._prepare_batch と同様だが、
        追加でグラフデータと commodity_list を含む。

        Returns:
            batch_data: {
                'x_nodes': Tensor [1, V, C] - torch.long,
                'x_commodities': Tensor [1, C, 3] - torch.float,
                'x_edges_capacity': Tensor [1, V, V] - torch.float,
                'x_edges': Tensor [1, V, V] - torch.long,
                'load_factor': Tensor [1] - torch.float (正解),
            }
        """
        ...

    def _train_epoch(self, train_loader, epoch: int) -> Dict:
        """1エポックの学習。"""
        ...

    def _validate_epoch(self, val_loader, epoch: int) -> Dict:
        """1エポックの検証。"""
        ...

    def _save_checkpoint(self, epoch: int, train_metrics: Dict,
                         val_metrics: Dict, is_best: bool = False) -> None:
        """チェックポイント保存。"""
        ...

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """チェックポイント読み込み。"""
        ...

    def _log_epoch(self, epoch: int, train_metrics: Dict,
                   val_metrics: Dict, lr: float, epoch_time: float) -> None:
        """エポックログ。"""
        ...

    def _save_training_log(self, total_training_time: float) -> None:
        """学習ログ保存。"""
        ...
```

**メトリクス** (SeqFlowRL と同一 + ILS 固有):

| メトリクス | 定義 | 備考 |
|---|---|---|
| Load Factor | `max_edge(usage / capacity)` | 全エッジの最大負荷率 |
| Complete Rate | 常に 100% | パスプール保証 |
| Complete Sample Rate | 常に 100% | パスプール保証 |
| Approximation Ratio | `mean(gt_lf / model_lf) * 100` | 最適解比 |
| Improvement | `(initial_lf - final_lf) / initial_lf * 100` | ILS改善率 |
| Num Iterations | ILS ループの実行ステップ数 | 効率指標 |

---

## 7. 設定ファイル

```json
// configs/gnn_ils/gnn_ils_base.json
{
    "_comment_meta": "=== GNN-ILS: GNN-guided Iterated Local Search ===",
    "expt_name": "gnn_ils_base",
    "gpu_id": "0",
    "use_gpu": false,

    "_comment_data": "=== Data Configuration ===",
    "solver_type": "pulp",
    "solver_time_limit": 60,
    "require_optimal": true,
    "graph_model": "random",
    "num_train_data": 3200,
    "num_val_data": 320,
    "num_test_data": 320,
    "num_nodes": 14,
    "num_commodities": 5,
    "sample_size": 5,
    "capacity_lower": 1000,
    "capacity_higher": 10000,
    "demand_lower": 5,
    "demand_higher": 500,

    "_comment_model": "=== Model Architecture ===",
    "hidden_dim": 128,
    "num_layers": 8,
    "aggregation": "mean",
    "dropout_rate": 0.3,

    "_comment_selector": "=== Selector Networks ===",
    "commodity_selector_mlp_layers": 2,
    "path_selector_mlp_layers": 2,
    "value_head_mlp_layers": 3,
    "use_graph_embedding_value": false,

    "_comment_path_pool": "=== Path Pool Configuration ===",
    "K": 10,
    "max_disjoint": 5,
    "max_candidate_paths": 15,

    "_comment_ils": "=== ILS Configuration ===",
    "max_iterations": 50,
    "no_improve_patience": 10,
    "perturbation_prob": 0.1,

    "_comment_rl": "=== RL Algorithm (A2C) ===",
    "entropy_weight": 0.01,
    "value_loss_weight": 0.5,
    "gamma": 0.99,
    "normalize_advantages": true,
    "grad_clip_norm": 1.0,
    "reward_mode": "shared",
    "_reward_mode_options": ["shared", "decomposed"],

    "_comment_training": "=== Training Configuration ===",
    "max_epochs": 50,
    "val_every": 5,
    "batch_size": 1,
    "_batch_size_note": "ILS は1サンプルずつ改善ループを回す",
    "samples_per_epoch": 100,
    "_samples_per_epoch_note": "1エポックあたりの処理サンプル数",
    "learning_rate": 0.0005,
    "decay_rate": 1.2,
    "lr_scheduler": "step",
    "weight_decay": 0.0001,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,

    "_comment_logging": "=== Logging and Checkpointing ===",
    "log_every": 10,
    "save_every": 10,
    "save_best_only": true,
    "tensorboard_dir": "runs/gnn_ils/",

    "_comment_eval": "=== Evaluation ===",
    "eval_deterministic": true,

    "_comment_debug": "=== Debug Options ===",
    "debug_mode": false,
    "verbose": true
}
```

---

## 8. エントリーポイント

### 8.1 学習スクリプト

```python
# scripts/gnn_ils/train_gnn_ils.py

"""
GNN-ILS Training Script.

Usage:
    python scripts/gnn_ils/train_gnn_ils.py --config configs/gnn_ils/gnn_ils_base.json
"""

import argparse
import torch

from src.common.config.config_manager import ConfigManager
from src.common.data_management.dataset_reader import DatasetReader
from src.gnn_ils.training.trainer import GNNILSTrainer


def main():
    parser = argparse.ArgumentParser(description='Train GNN-ILS')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # 設定読み込み
    config = ConfigManager.load(args.config)

    # データローダー (batch_size=1)
    train_loader = DatasetReader(config, 'train')
    val_loader = DatasetReader(config, 'val')

    # トレーナー
    trainer = GNNILSTrainer(config)

    # 学習
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
```

### 8.2 テストスクリプト

```python
# scripts/gnn_ils/test_gnn_ils.py

"""
GNN-ILS Test Script.

Usage:
    python scripts/gnn_ils/test_gnn_ils.py --config configs/gnn_ils/gnn_ils_base.json
"""

import argparse
import torch

from src.common.config.config_manager import ConfigManager
from src.common.data_management.dataset_reader import DatasetReader
from src.gnn_ils.training.trainer import GNNILSTrainer


def main():
    parser = argparse.ArgumentParser(description='Test GNN-ILS')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # 設定読み込み
    config = ConfigManager.load(args.config)

    # テストデータ
    test_loader = DatasetReader(config, 'test')

    # トレーナー & チェックポイント読み込み
    trainer = GNNILSTrainer(config)
    trainer.load_checkpoint(config.get('load_model_path', 'saved_models/gnn_ils/best_model.pt'))

    # テスト
    test_metrics = trainer._validate_epoch(test_loader, epoch=0)

    # 結果表示
    print(f"\n{'='*70}")
    print(f"TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Load Factor:        {test_metrics['mean_load_factor']:.4f}")
    print(f"  Complete Rate:      {test_metrics['complete_rate']:.1f}%")
    print(f"  Approx Ratio:       {test_metrics.get('approximation_ratio', 'N/A')}")
    print(f"  Mean Improvement:   {test_metrics.get('improvement', 0):.2f}%")
    print(f"  Mean Iterations:    {test_metrics.get('num_iterations', 0):.1f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
```

---

## 9. 再利用コンポーネント一覧

| コンポーネント | 元ファイル | 再利用方法 |
|---|---|---|
| `ResidualGatedGCNLayer` | `src/gcn/models/gcn_layers.py` | import して直接使用 |
| `BatchNormNode` | `src/gcn/models/gcn_layers.py` | import して直接使用 |
| `BatchNormEdge` | `src/gcn/models/gcn_layers.py` | import して直接使用 |
| `MLP` | `src/gcn/models/gcn_layers.py` | import して直接使用 |
| `KShortestPathFinder` | `src/common/graph/k_shortest_path.py` | PathPoolManager 内で使用 |
| `ConfigManager` | `src/common/config/config_manager.py` | エントリーポイントで使用 |
| `DatasetReader` | `src/common/data_management/dataset_reader.py` | データ読み込みに使用 |
| `validate_batch_types` | `src/common/types.py` | バッチ検証に使用 |
| `validate_encoder_input_types` | `src/common/types.py` | エンコーダ入力検証に使用 |
| `get_model_root` | `src/common/config/paths.py` | モデル保存パス取得 |
| `get_graph_file` | `src/common/config/paths.py` | グラフファイルパス取得 |
| `get_commodity_file` | `src/common/config/paths.py` | コモディティファイルパス取得 |

---

## 10. 新規実装一覧

| ファイル | クラス/関数 | 概要 |
|---|---|---|
| `src/gnn_ils/models/gnn_ils_encoder.py` | `GNNILSEncoder` | GNNエンコーダ (HybridGNNEncoder と同一構造) |
| `src/gnn_ils/models/commodity_selector.py` | `CommoditySelector` | Level1: コモディティ選択ポリシー |
| `src/gnn_ils/models/path_selector.py` | `PathSelector` | Level2: パス選択ポリシー |
| `src/gnn_ils/models/value_head.py` | `ILSValueHead` | Critic (graph_embedding版も対応) |
| `src/gnn_ils/models/gnn_ils_model.py` | `GNNILSModel` | 統合 Actor-Critic モデル |
| `src/gnn_ils/environment/path_pool_manager.py` | `PathPoolManager` | 候補パスプール構築 (KSP + disjoint) |
| `src/gnn_ils/environment/ils_environment.py` | `ILSEnvironment` | ILS改善ループ環境 |
| `src/gnn_ils/training/ils_a2c_strategy.py` | `ILSA2CStrategy` | 2段階A2C損失計算 |
| `src/gnn_ils/training/trainer.py` | `GNNILSTrainer` | 学習・検証・テストループ |
| `src/gnn_ils/utils/load_utils.py` | `compute_load_factor`, `compute_edge_usage` | 負荷計算ユーティリティ |
| `scripts/gnn_ils/train_gnn_ils.py` | `main()` | 学習エントリーポイント |
| `scripts/gnn_ils/test_gnn_ils.py` | `main()` | テストエントリーポイント |
| `configs/gnn_ils/gnn_ils_base.json` | — | 基本設定ファイル |

---

## 11. 実装順序

依存関係を考慮した6フェーズの実装順序。

### Phase 1: ユーティリティ & パスプール

**依存関係**: なし (既存コンポーネントのみ使用)

1. `src/gnn_ils/__init__.py` (空ファイル群)
2. `src/gnn_ils/utils/load_utils.py` — 負荷計算関数
3. `src/gnn_ils/environment/path_pool_manager.py` — KSP + link-disjoint パスプール

**検証**: パスプール構築の単体テスト (正しいパス数、重複排除、全コモディティ到達)

### Phase 2: エンコーダ

**依存関係**: Phase 1 (テスト用に load_utils を使用)

4. `src/gnn_ils/models/gnn_ils_encoder.py` — GNNILSEncoder

**検証**: ダミー入力でフォワードパス成功、出力形状の確認

### Phase 3: ポリシーネットワーク

**依存関係**: Phase 2 (encoder の出力を入力に使用)

5. `src/gnn_ils/models/commodity_selector.py` — CommoditySelector
6. `src/gnn_ils/models/path_selector.py` — PathSelector
7. `src/gnn_ils/models/value_head.py` — ILSValueHead
8. `src/gnn_ils/models/gnn_ils_model.py` — GNNILSModel

**検証**: 各ヘッドの出力形状、確率分布の合計=1、マスクの動作確認

### Phase 4: 環境

**依存関係**: Phase 1 (PathPoolManager, load_utils)

9. `src/gnn_ils/environment/ils_environment.py` — ILSEnvironment

**検証**: reset → step ループの動作、負荷計算の正確性、到達率100%の確認

### Phase 5: 学習戦略 & トレーナー

**依存関係**: Phase 3 (GNNILSModel), Phase 4 (ILSEnvironment)

10. `src/gnn_ils/training/ils_a2c_strategy.py` — ILSA2CStrategy
11. `src/gnn_ils/training/trainer.py` — GNNILSTrainer

**検証**: 1エピソードの train_step 実行、損失の逆伝播、メトリクス収集

### Phase 6: エントリーポイント & 設定

**依存関係**: Phase 5

12. `configs/gnn_ils/gnn_ils_base.json`
13. `scripts/gnn_ils/train_gnn_ils.py`
14. `scripts/gnn_ils/test_gnn_ils.py`

**検証**: エンドツーエンドの学習・テスト実行 (小規模データ)

---

## 補足: 設計判断のまとめ

| 判断事項 | 選択 | 理由 |
|---|---|---|
| エンコーダ | 新規作成 (HybridGNNEncoder と同一構造) | ILS 固有の入力特徴量 (毎ステップ更新される edge_usage) に対応 |
| ResidualGatedGCNLayer | 直接再利用 | 層の構造は共通 |
| Link-disjoint paths | `nx.edge_disjoint_paths` で新規実装 | 既存 KShortestPathFinder にはない機能 |
| Gymnasium | 不使用 (独自ループ) | テンソル状態空間が Box/Discrete に収まらない |
| batch_size | 1 | ILS は1サンプルずつ改善ループを回す |
| 報酬 | 改善ベース `-(new_lf - old_lf)` | RL-KSP と同じ設計思想 |
| 到達率保証 | パスプール制約 | パスプール内の有効パスのみ選択可能 |
| エンコーダ共有 | Actor/Critic 共有 | SeqFlowRL と同じ設計 |
| Value 関数 | グローバル V(s) | 目的関数がグローバルな最大負荷率のため |
