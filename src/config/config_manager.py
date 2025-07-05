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
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        
        if torch.cuda.is_available():
            print(f"CUDA available, using GPU ID {self.config.gpu_id}")
            dtypeFloat = torch.float32
            dtypeLong = torch.long
            torch.cuda.manual_seed(1)
        else:
            print("CUDA not available")
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