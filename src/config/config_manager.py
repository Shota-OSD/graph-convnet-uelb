import os
import torch
from ..config import get_config

class ConfigManager:
    """設定とGPUの管理を担当するクラス"""
    
    def __init__(self, config_path: str = "configs/default2.json"):
        self.config_path = config_path
        self.config = get_config(config_path)
        self.dtypeFloat, self.dtypeLong = self._setup_gpu()
    
    def _setup_gpu(self):
        """GPU設定を行う"""
        use_gpu = getattr(self.config, 'use_gpu', True)  # デフォルトはTrue
        
        if use_gpu and torch.cuda.is_available():
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
            print(f"CUDA available, using GPU ID {self.config.gpu_id}")
            dtypeFloat = torch.float32
            dtypeLong = torch.long
            torch.cuda.manual_seed(1)
        else:
            if use_gpu and not torch.cuda.is_available():
                print("GPU requested but CUDA not available, falling back to CPU")
            else:
                print("Using CPU (GPU disabled in config)")
            dtypeFloat = torch.float32
            dtypeLong = torch.long
            torch.manual_seed(1)
        
        return dtypeFloat, dtypeLong
    
    def get_config(self):
        """設定を取得"""
        return self.config
    
    def get_dtypes(self):
        """データ型を取得"""
        return self.dtypeFloat, self.dtypeLong 