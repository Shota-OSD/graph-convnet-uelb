"""データセットとモデル保管場所の解決ヘルパー.

データセットと学習済みモデルはコードリポジトリ外で管理する。
パスの解決優先順位:

1. 環境変数 (``GCN_UELB_DATA_ROOT`` / ``GCN_UELB_MODEL_ROOT``)
2. config の ``data_root`` / ``model_root``
3. デフォルト (プロジェクトルートと同階層の ``../ml-data/graph-convnet-uelb/``)

ディレクトリ構造::

    <Projects>/ml-data/graph-convnet-uelb/
    ├── datasets/
    │   └── {dataset_name}/
    │       ├── train_data/
    │       ├── val_data/
    │       └── test_data/
    └── saved_models/

``dataset_name`` は config の ``dataset_name`` フィールド、なければ ``expt_name`` にフォールバック。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

ENV_DATA_ROOT = "GCN_UELB_DATA_ROOT"
ENV_MODEL_ROOT = "GCN_UELB_MODEL_ROOT"

# プロジェクトルート (このファイルから 4 階層上: src/common/config/paths.py → repo)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# プロジェクトと同階層の ml-data/graph-convnet-uelb をデフォルトに
_DEFAULT_BASE = _PROJECT_ROOT.parent / "ml-data" / "graph-convnet-uelb"
DEFAULT_DATA_ROOT = _DEFAULT_BASE / "datasets"
DEFAULT_MODEL_ROOT = _DEFAULT_BASE / "saved_models"


def _config_get(config: Any, key: str) -> Optional[Any]:
    if config is None:
        return None
    if hasattr(config, "get"):
        try:
            return config.get(key)
        except TypeError:
            pass
    return getattr(config, key, None)


def get_data_root(config: Any = None) -> Path:
    """データセットを格納する親ディレクトリを返す."""
    env_val = os.environ.get(ENV_DATA_ROOT)
    if env_val:
        return Path(env_val).expanduser()
    cfg_val = _config_get(config, "data_root")
    if cfg_val:
        return Path(str(cfg_val)).expanduser()
    return DEFAULT_DATA_ROOT


def get_model_root(config: Any = None) -> Path:
    """学習済みモデルを格納する親ディレクトリを返す."""
    env_val = os.environ.get(ENV_MODEL_ROOT)
    if env_val:
        return Path(env_val).expanduser()
    cfg_val = _config_get(config, "model_root")
    if cfg_val:
        return Path(str(cfg_val)).expanduser()
    return DEFAULT_MODEL_ROOT


def get_dataset_name(config: Any) -> str:
    """config に紐付くデータセット名を返す.

    ``dataset_name`` が無ければ ``expt_name`` にフォールバック。
    """
    name = _config_get(config, "dataset_name") or _config_get(config, "expt_name")
    if not name:
        raise ValueError(
            "Config must specify 'dataset_name' (or 'expt_name' as fallback) "
            "to locate the dataset directory."
        )
    return str(name)


def get_dataset_dir(config: Any) -> Path:
    """config に紐付くデータセットのルートディレクトリを返す."""
    return get_data_root(config) / get_dataset_name(config)


def get_mode_dir(mode: str, config: Any) -> Path:
    """``train_data`` / ``val_data`` / ``test_data`` のディレクトリを返す."""
    return get_dataset_dir(config) / f"{mode}_data"


def get_exact_solution_file(mode: str, config: Any) -> Path:
    """モード毎の exact_solution.csv のパスを返す."""
    return get_mode_dir(mode, config) / "exact_solution.csv"


def get_graph_file(mode: str, idx: int, config: Any) -> Path:
    """個別グラフファイルのパスを返す (10件ごとに番号付けされたサブディレクトリ)."""
    bucket = idx - (idx % 10)
    return get_mode_dir(mode, config) / "graph_file" / str(bucket) / f"graph_{idx}.gml"


def get_commodity_file(mode: str, idx: int, config: Any) -> Path:
    bucket = idx - (idx % 10)
    return (
        get_mode_dir(mode, config)
        / "commodity_file"
        / str(bucket)
        / f"commodity_data_{idx}.csv"
    )


def get_node_flow_file(mode: str, idx: int, config: Any) -> Path:
    bucket = idx - (idx % 10)
    return (
        get_mode_dir(mode, config)
        / "node_flow_file"
        / str(bucket)
        / f"node_flow_{idx}.csv"
    )


def dataset_exists(config: Any, modes: tuple = ("train", "val", "test")) -> bool:
    """指定 config のデータセットが存在するかを確認する.

    各モードディレクトリと必要なサブディレクトリ (commodity_file,
    graph_file, node_flow_file) の存在をチェック。
    """
    required_subdirs = ("commodity_file", "graph_file", "node_flow_file")
    for mode in modes:
        mode_dir = get_mode_dir(mode, config)
        if not mode_dir.exists():
            return False
        for sub in required_subdirs:
            sub_dir = mode_dir / sub
            if not sub_dir.exists():
                return False
            # 数値名のサブディレクトリ (0, 10, 20, ...) が1つ以上あるか
            numeric_subs = [
                d for d in sub_dir.iterdir() if d.is_dir() and d.name.isdigit()
            ]
            if not numeric_subs:
                return False
    return True
