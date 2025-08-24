#!/usr/bin/env python3
"""
モデル保存・読み込み機能のテスト
"""

import torch
import json
import os
import shutil
from src.config.config_manager import ConfigManager
from src.train.trainer import Trainer

def test_model_saving_loading():
    """モデル保存・読み込みのテスト"""
    print("Testing model saving and loading functionality...")
    
    # テスト用の設定を作成
    test_config = {
        "expt_name": "test_model_persistence",
        "gpu_id": "0",
        "use_gpu": False,  # テスト用にCPUを使用
        
        "num_train_data": 10,
        "num_test_data": 5,
        "num_val_data": 5,
        "num_nodes": 10,
        "num_commodities": 3,
        "sample_size": 2,
        "capacity_lower": 100,
        "capacity_higher": 1000,
        "demand_lower": 1,
        "demand_higher": 50,
        
        "node_dim": 4,
        "voc_nodes_in": 10,
        "voc_nodes_out": 2,
        "voc_edges_in": 3,
        "voc_edges_out": 2,
        
        "beam_size": 5,
        "hidden_dim": 16,
        "num_layers": 2,
        "mlp_layers": 1,
        "aggregation": "mean",
        
        "max_epochs": 2,
        "val_every": 1,
        "test_every": 1,
        
        "batch_size": 2,
        "batches_per_epoch": 2,
        "accumulation_steps": 1,
        
        "learning_rate": 0.001,
        "decay_rate": 1.0,
        "dropout_rate": 0.1,
        
        "models_dir": "./test_saved_models",
        "save_model": True,
        "save_every_epoch": True,
        "load_saved_model": False,
        "cleanup_old_models": False
    }
    
    # テスト用設定ファイルを作成
    os.makedirs("./test_configs", exist_ok=True)
    test_config_path = "./test_configs/test_model_persistence.json"
    with open(test_config_path, 'w') as f:
        json.dump(test_config, f, indent=4)
    
    try:
        # クリーンアップ
        if os.path.exists("./test_saved_models"):
            shutil.rmtree("./test_saved_models")
        
        # Step 1: 設定を読み込んでトレーナーを作成
        print("\n1. Creating trainer with test config...")
        config_manager = ConfigManager(test_config_path)
        config = config_manager.get_config()
        dtypeFloat, dtypeLong = config_manager.get_dtypes()
        
        trainer1 = Trainer(config, dtypeFloat, dtypeLong)
        
        # Step 2: モデルのパラメータを取得（保存前）
        print("\n2. Getting initial model parameters...")
        initial_params = {}
        for name, param in trainer1.get_model().named_parameters():
            initial_params[name] = param.data.clone()
        
        # Step 3: モデルを保存
        print("\n3. Saving model...")
        trainer1.save_model(epoch=0, loss=0.5)
        
        # 保存されたファイルが存在することを確認
        models_dir = "./test_saved_models"
        assert os.path.exists(models_dir), f"Models directory {models_dir} not created"
        
        saved_files = os.listdir(models_dir)
        print(f"Saved files: {saved_files}")
        assert len(saved_files) > 0, "No model files were saved"
        
        # Step 4: 新しいトレーナーを作成して読み込み設定を有効にする
        print("\n4. Creating new trainer with load_saved_model=True...")
        config.load_saved_model = True
        trainer2 = Trainer(config, dtypeFloat, dtypeLong)
        
        # Step 5: パラメータが正しく読み込まれたかチェック
        print("\n5. Verifying loaded parameters...")
        loaded_params = {}
        for name, param in trainer2.get_model().named_parameters():
            loaded_params[name] = param.data.clone()
        
        # パラメータの比較
        params_match = True
        for name in initial_params:
            if not torch.equal(initial_params[name], loaded_params[name]):
                params_match = False
                print(f"Parameter mismatch: {name}")
                break
        
        if params_match:
            print("✅ Model parameters loaded correctly!")
        else:
            print("❌ Model parameters do not match!")
            return False
        
        # Step 6: モデルファイルの削除テスト
        print("\n6. Testing model cleanup...")
        # いくつかのエポックでモデルを保存
        for epoch in range(1, 6):
            trainer1.save_model(epoch=epoch, loss=0.4-epoch*0.05)
        
        files_before_cleanup = len(os.listdir(models_dir))
        print(f"Files before cleanup: {files_before_cleanup}")
        
        # 古いモデルを削除（最新3個を保持）
        trainer1.cleanup_old_models(keep_last_n=3)
        
        files_after_cleanup = len(os.listdir(models_dir))
        print(f"Files after cleanup: {files_after_cleanup}")
        
        # Step 7: コンフィグハッシュのテスト
        print("\n7. Testing config hash consistency...")
        hash1 = trainer1._get_config_hash()
        hash2 = trainer2._get_config_hash()
        
        if hash1 == hash2:
            print(f"✅ Config hashes match: {hash1}")
        else:
            print(f"❌ Config hash mismatch: {hash1} vs {hash2}")
            return False
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # クリーンアップ
        if os.path.exists("./test_saved_models"):
            shutil.rmtree("./test_saved_models")
        if os.path.exists("./test_configs"):
            shutil.rmtree("./test_configs")

if __name__ == "__main__":
    success = test_model_saving_loading()
    if success:
        print("\n🎉 Model persistence functionality is working correctly!")
    else:
        print("\n💥 Model persistence functionality has issues!")
        exit(1)