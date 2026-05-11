import pytest
import torch

from src.gnn_ils.training.return_normalizer import RunningReturnNormalizer


class TestRunningReturnNormalizer:
    """RunningReturnNormalizer のユニットテスト。"""

    def test_first_call_normalizes_to_zero_mean_unit_std(self):
        """初回呼び出しでバッチ統計を使い mean~0, std~1 に正規化される。"""
        normalizer = RunningReturnNormalizer()
        returns_t = torch.tensor([10.0, 20.0, 30.0])

        result = normalizer.normalize(returns_t)

        assert abs(result.mean().item()) < 1e-5
        assert abs(result.std().item() - 1.0) < 0.2  # 3要素なので厳密に1にはならない
        assert normalizer._initialized is True

    def test_running_stats_update(self):
        """2回目の呼び出しで running_mean/var が momentum で加重平均更新される。"""
        normalizer = RunningReturnNormalizer(momentum=0.99)

        first_returns = torch.tensor([10.0, 20.0, 30.0])
        normalizer.normalize(first_returns)
        first_mean = first_returns.mean().item()
        first_var = first_returns.var().item()

        second_returns = torch.tensor([100.0, 200.0, 300.0])
        normalizer.normalize(second_returns)
        second_mean = second_returns.mean().item()
        second_var = second_returns.var().item()

        expected_mean = 0.99 * first_mean + 0.01 * second_mean
        expected_var = 0.99 * first_var + 0.01 * second_var

        assert abs(normalizer.running_mean - expected_mean) < 1e-5
        assert abs(normalizer.running_var - expected_var) < 1e-5

    def test_constant_returns_no_nan(self):
        """分散 0 の入力でも nan/inf が発生しない。"""
        normalizer = RunningReturnNormalizer()
        returns_t = torch.tensor([5.0, 5.0, 5.0])

        result = normalizer.normalize(returns_t)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_single_step_episode(self):
        """要素数 1 のテンソルでもエラーなく正規化される。"""
        normalizer = RunningReturnNormalizer()
        returns_t = torch.tensor([3.0])

        result = normalizer.normalize(returns_t)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert result.shape == returns_t.shape

    def test_reset(self):
        """reset() で内部状態が初期値に戻り、次の normalize が初回と同じ挙動になる。"""
        normalizer = RunningReturnNormalizer()

        # 数回 normalize を呼ぶ
        normalizer.normalize(torch.tensor([1.0, 2.0, 3.0]))
        normalizer.normalize(torch.tensor([10.0, 20.0, 30.0]))
        normalizer.normalize(torch.tensor([100.0, 200.0, 300.0]))

        normalizer.reset()

        assert normalizer._initialized is False
        assert normalizer.running_mean == 0.0
        assert normalizer.running_var == 1.0

        # reset 後の normalize は初回と同じ挙動（バッチ統計で初期化）
        returns_t = torch.tensor([10.0, 20.0, 30.0])
        result = normalizer.normalize(returns_t)

        assert normalizer._initialized is True
        assert abs(normalizer.running_mean - returns_t.mean().item()) < 1e-5
        assert abs(result.mean().item()) < 1e-5

    def test_normalize_returns_flag_disabled(self):
        """normalize を呼ばなければ _initialized は False のまま。"""
        normalizer = RunningReturnNormalizer()

        assert normalizer._initialized is False
        assert normalizer.running_mean == 0.0
        assert normalizer.running_var == 1.0

    def test_output_shape_preserved(self):
        """入力と出力の shape が一致する (T=1, 5, 50)。"""
        normalizer = RunningReturnNormalizer()

        for t in [1, 5, 50]:
            normalizer.reset()
            returns_t = torch.randn(t)
            result = normalizer.normalize(returns_t)
            assert result.shape == returns_t.shape, f"T={t} で shape 不一致"

    def test_momentum_effect(self):
        """momentum が小さいほど running_mean の更新が速い。"""
        normalizer_slow = RunningReturnNormalizer(momentum=0.99)
        normalizer_fast = RunningReturnNormalizer(momentum=0.5)

        batches = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([100.0, 200.0, 300.0]),
            torch.tensor([1000.0, 2000.0, 3000.0]),
        ]

        for batch in batches:
            normalizer_slow.normalize(batch)
            normalizer_fast.normalize(batch)

        last_batch_mean = batches[-1].mean().item()

        # momentum=0.5 の方が直近バッチの平均に近い（更新が速い）
        diff_slow = abs(normalizer_slow.running_mean - last_batch_mean)
        diff_fast = abs(normalizer_fast.running_mean - last_batch_mean)

        assert diff_fast < diff_slow, (
            f"momentum=0.5 の方が直近バッチに近いはず: "
            f"diff_fast={diff_fast}, diff_slow={diff_slow}"
        )
