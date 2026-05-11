import torch


class RunningReturnNormalizer:
    """
    Return (discounted cumulative reward) の Running Normalization。

    エピソードをまたいで return の mean/var を指数移動平均で追跡し、
    各エピソードの return を正規化する。Critic の学習ターゲットを
    reward_scale に依存しない安定したスケールに保つ。

    使い方:
        normalizer = RunningReturnNormalizer()
        returns_t = compute_returns(rewards, gamma)
        returns_normalized = normalizer.normalize(returns_t)
        critic_loss = F.mse_loss(state_values, returns_normalized)
    """

    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-8):
        """
        Args:
            momentum: 指数移動平均の係数。1に近いほど過去の統計を重視。
                      0.99 = 直近100エピソード程度の統計を反映。
            epsilon: ゼロ除算防止の微小値。
        """
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = 0.0
        self.running_var = 1.0
        self._initialized = False

    def normalize(self, returns_t: torch.Tensor) -> torch.Tensor:
        """
        Return テンソルを正規化する。

        初回呼び出し時はバッチ統計で初期化。以降は指数移動平均で更新。

        Args:
            returns_t: [T] 形状の return テンソル（1エピソード分）

        Returns:
            正規化された return テンソル [T]（mean≈0, std≈1）
        """
        with torch.no_grad():
            batch_mean = returns_t.mean().item()
            batch_var = returns_t.var().item() if returns_t.numel() > 1 else 1.0

            if not self._initialized:
                self.running_mean = batch_mean
                self.running_var = max(batch_var, self.epsilon)
                self._initialized = True
            else:
                self.running_mean = (
                    self.momentum * self.running_mean
                    + (1.0 - self.momentum) * batch_mean
                )
                self.running_var = (
                    self.momentum * self.running_var
                    + (1.0 - self.momentum) * batch_var
                )

        return (returns_t - self.running_mean) / (
            (self.running_var ** 0.5) + self.epsilon
        )

    def reset(self):
        """統計をリセットする（新規学習開始時に使用）。"""
        self.running_mean = 0.0
        self.running_var = 1.0
        self._initialized = False
