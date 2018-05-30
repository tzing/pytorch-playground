import pytest
import torch
from torch.autograd import Variable

from functional.iou import iou


@pytest.mark.parametrize('n_base, base_dim', [
    (0, (0,)),
    (1, (4,)),
    (2, (2,4)),
    (6, (2,3,4))
])
@pytest.mark.parametrize('n_target, target_dim', [
    (0, (0,)),
    (1, (4,)),
    (3, (3,4)),
    (5, (1,5,4))
])
def test_iou(n_base, base_dim, n_target, target_dim):
    base = torch.rand(*base_dim)
    target = torch.rand(*target_dim)

    score = iou(base, target)

    if n_base * n_target == 0:
        assert len(score) == 0
    else:
        assert score.size() == (n_base, n_target)
        assert score.max() <= 1
        assert score.min() >= 0


def test_iou_value():
    # values
    base = torch.tensor([
        [50, 50, 10, 10],
        [55, 53, 10, 12],
        [50, 51, 4, 2]
    ]).float()
    target = torch.tensor([
        [50, 50, 10, 10],
        [51, 51, 2, 2],
        [48, 48, 6, 1],
        [5, 5, 1, 1]
    ]).float()

    # run
    score = iou(base, target)

    # test
    assert score.size() == (3, 4)
    assert torch.max(score) <= 1
    assert torch.min(score) >= 0

    ans = torch.tensor([
        [1.0000, 0.0400, 0.0600, 0.0000],
        [0.2222, 0.0333, 0.0080, 0.0000],
        [0.0800, 0.5000, 0.0000, 0.0000]
    ])
    assert (score - ans).abs().max() < 1e-4
