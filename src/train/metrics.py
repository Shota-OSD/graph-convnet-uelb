import numpy as np
from typing import List, Tuple

class MetricsLogger:
    """メトリクスの記録と出力を管理するクラス"""
    
    def __init__(self):
        self.train_loss_list = []
        self.train_err_edges_list = []
        self.val_approximation_rate_list = []
        self.test_approximation_rate_list = []
    
    def log_train_metrics(self, loss: float, edge_error: float):
        """トレーニングメトリクスを記録"""
        self.train_loss_list.append(loss)
        self.train_err_edges_list.append(edge_error)
    
    def log_val_metrics(self, approximation_rate: float):
        """検証メトリクスを記録"""
        self.val_approximation_rate_list.append(approximation_rate)
    
    def log_test_metrics(self, approximation_rate: float):
        """テストメトリクスを記録"""
        self.test_approximation_rate_list.append(approximation_rate)
    
    def get_final_metrics(self) -> dict:
        """最終メトリクスを取得"""
        return {
            'final_train_loss': self.train_loss_list[-1] if self.train_loss_list else 0.0,
            'final_edge_error': self.train_err_edges_list[-1] if self.train_err_edges_list else 0.0,
            'final_val_approximation_rate': self.val_approximation_rate_list[-1] if self.val_approximation_rate_list else 0.0,
            'final_test_approximation_rate': self.test_approximation_rate_list[-1] if self.test_approximation_rate_list else 0.0,
            'best_val_approximation_rate': max(self.val_approximation_rate_list) if self.val_approximation_rate_list else 0.0,
            'best_test_approximation_rate': max(self.test_approximation_rate_list) if self.test_approximation_rate_list else 0.0
        }
    
    def print_summary(self):
        """最終結果のサマリーを表示"""
        metrics = self.get_final_metrics()
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Final Train Loss: {metrics['final_train_loss']:.4f}")
        print(f"Final Train Edge Error: {metrics['final_edge_error']:.2f}%")
        if self.val_approximation_rate_list:
            print(f"Final Val Approximation Rate: {metrics['final_val_approximation_rate']:.2f}%")
        if self.test_approximation_rate_list:
            print(f"Final Test Approximation Rate: {metrics['final_test_approximation_rate']:.2f}%")
    
    def print_results_table(self, val_every: int = 3, test_every: int = 3):
        """結果を表形式で出力"""
        print("\n" + "="*100)
        print("TRAINING RESULTS SUMMARY")
        print("="*100)
        
        # ヘッダー
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Edge Error':<12} {'Val Approx':<12} {'Test Approx':<12}")
        print("-" * 60)
        
        # データ行
        max_epochs = len(self.train_loss_list)
        for epoch in range(max_epochs):
            train_loss = self.train_loss_list[epoch]
            edge_error = self.train_err_edges_list[epoch]
            
            # 検証とテストの近似率（該当するエポックのみ）
            val_approx = "N/A"
            test_approx = "N/A"
            
            # 検証近似率の取得
            val_epochs = list(range(0, max_epochs, val_every)) + [max_epochs - 1]
            if epoch in val_epochs and len(self.val_approximation_rate_list) > 0:
                val_idx = val_epochs.index(epoch)
                if val_idx < len(self.val_approximation_rate_list):
                    val_approx = f"{self.val_approximation_rate_list[val_idx]:.2f}%"
            
            # テスト近似率の取得
            test_epochs = list(range(0, max_epochs, test_every)) + [max_epochs - 1]
            if epoch in test_epochs and len(self.test_approximation_rate_list) > 0:
                test_idx = test_epochs.index(epoch)
                if test_idx < len(self.test_approximation_rate_list):
                    test_approx = f"{self.test_approximation_rate_list[test_idx]:.2f}%"
            
            print(f"{epoch+1:<6} {train_loss:<12.4f} {edge_error:<12.2f} {val_approx:<12} {test_approx:<12}")
        
        print("-" * 60)
        
        # 最終結果のサマリー
        metrics = self.get_final_metrics()
        print(f"\nFinal Results:")
        print(f"  Final Train Loss: {metrics['final_train_loss']:.4f}")
        print(f"  Final Edge Error: {metrics['final_edge_error']:.2f}%")
        if self.val_approximation_rate_list:
            print(f"  Best Val Approximation Rate: {metrics['best_val_approximation_rate']:.2f}%")
        if self.test_approximation_rate_list:
            print(f"  Best Test Approximation Rate: {metrics['best_test_approximation_rate']:.2f}%")

def metrics_to_str(epoch: int, time: float, learning_rate: float, loss: float, 
                  mean_maximum_load_factor: float, gt_load_factor: float, 
                  approximation_rate: float, infeasible_rate: float) -> str:
    """メトリクスを文字列形式で返す関数"""
    return (f'epoch:{epoch:02d}\t'
            f'execution time:{time:.8f}s\t'
            f'lr:{learning_rate:.2e}\t'
            f'loss:{loss:.4f}\t'
            f'mean_maximum_load_factor:{mean_maximum_load_factor:.3f}\t'
            f'gt_load_factor:{gt_load_factor:.3f}\t'
            f'approximation_rate:{approximation_rate:.3f}\t'
            f'infeasible_rate:{infeasible_rate:.3f}') 