#!/bin/bash

# Graph ConvNet UELB Environment Setup Script
# このスクリプトは conda 環境を自動でセットアップします

set -e  # エラーが発生したら終了

echo "🚀 Graph ConvNet UELB 環境のセットアップを開始します..."

# conda が利用可能かチェック
if ! command -v conda &> /dev/null; then
    echo "❌ conda が見つかりません。Anaconda または Miniconda をインストールしてください。"
    exit 1
fi

# 環境名
ENV_NAME="gcn-uelb-env"

# 既存の環境をチェック
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "⚠️  環境 '${ENV_NAME}' が既に存在します。"
    read -p "削除して再作成しますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  既存の環境を削除しています..."
        conda env remove -n ${ENV_NAME}
    else
        echo "❌ セットアップをキャンセルしました。"
        exit 1
    fi
fi

echo "📦 conda環境 '${ENV_NAME}' を作成しています..."

# Python 3.7.9の環境を作成
conda create -n ${ENV_NAME} python=3.7.9 -y

echo "✅ 環境 '${ENV_NAME}' が作成されました。"

# 環境をアクティベート
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "🔧 基本パッケージをインストールしています..."

# PyTorchをインストール
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS の場合
    echo "🍎 macOS用のPyTorchをインストールしています..."
    conda install pytorch torchvision torchaudio -c pytorch -y
else
    # Linux の場合
    echo "🐧 Linux用のPyTorchをインストールしています..."
    # CUDA対応かチェック
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 CUDA対応版PyTorchをインストールしています..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo "💻 CPU版PyTorchをインストールしています..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
fi

# Jupyter Notebookをインストール
conda install jupyter -y

echo "📋 requirements.txtから依存関係をインストールしています..."

# requirements.txtの依存関係をインストール
pip install -r requirements.txt

echo "🔍 インストールを検証しています..."

# インストールの検証
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import networkx; print(f'NetworkX version: {networkx.__version__}')"

echo ""
echo "🎉 セットアップが完了しました！"
echo ""
echo "次のコマンドで環境をアクティベートしてください："
echo "  conda activate ${ENV_NAME}"
echo ""
echo "学習を開始するには："
echo "  python main.py"
echo ""
echo "または Jupyter Notebook で："
echo "  jupyter notebook main.ipynb"
echo ""

# libarchiveの問題の対処法を表示
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📝 macOSでlibarchiveエラーが発生する場合："
    echo "  brew install libarchive"
    echo "  ln -sf /opt/homebrew/opt/libarchive/lib/libarchive.dylib ~/miniconda3/lib/libarchive.19.dylib"
    echo ""
fi

echo "詳細な使い方はREADME.mdを参照してください。" 