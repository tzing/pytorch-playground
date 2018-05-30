import pytest
import torch

from functional.nms import nms


@pytest.mark.parametrize('n_box', [0, 1, 100])
@pytest.mark.parametrize('threshold', [0, .5, 1])
def test_nms(n_box, threshold):
    # generate data
    box = torch.rand(n_box, 4) * 10
    score = torch.rand(n_box)

    # test
    mbox, _, index, num = nms(box, score, threshold)

    # assert
    assert len(mbox) <= len(box)
    if n_box > 0:
        assert index.max() < len(mbox)
        assert index.min() >= 0
        assert num.max() <= len(box)
        assert num.min() > 0
    else:
        assert len(mbox) == 0
        assert len(index) == 0
        assert len(num) == 0
