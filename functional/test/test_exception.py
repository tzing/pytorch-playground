import torch
import pytest

from functional.exception import assert_last_dim, DimensionError


def test_assert_last_dim():
    assert_last_dim(torch.rand(5), 5, False)
    assert_last_dim(torch.rand(3, 5), 5, False)
    assert_last_dim(torch.rand(2, 4, 5), 5, False)


def test_assert_last_dim_allow_0d():
    assert_last_dim(torch.rand(0), 5, True)


def test_assert_last_dim_raise():
    with pytest.raises(DimensionError):
        assert_last_dim(torch.rand(0), 5, False)

    with pytest.raises(DimensionError):
        assert_last_dim(torch.rand(4), 5, False)

    with pytest.raises(DimensionError):
        assert_last_dim(torch.rand(3, 4), 5, False)

    with pytest.raises(DimensionError):
        assert_last_dim(torch.rand(2, 4, 6), 5, False)
