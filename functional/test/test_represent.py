import pytest
import torch

from functional.represent import roi2reg, reg2roi


def test_reg2roi():
    # data
    regr = torch.FloatTensor([
        [0, 0, 0, 0],
        [0, 0, 1, 1]
    ])
    acr = torch.FloatTensor([
        [50, 50, 50, 50]
    ]).expand(*regr.size())

    # calc
    roi = reg2roi(regr, acr)

    # ans
    ans = torch.FloatTensor([
        [50, 50, 50, 50],
        [50, 50, 135.914, 135.914]
    ])

    assert torch.max(torch.abs(roi - ans)) < 1E-4


@pytest.mark.parametrize('num_sample', [
    (0,),
    (4,),
    (3, 4),
    (2, 5, 4)
])
def test_reg2roi_cross(num_sample):
    orig_reg = torch.rand(*num_sample)
    anchor = torch.rand(*num_sample)

    roi = reg2roi(orig_reg, anchor)
    reg = roi2reg(roi, anchor)

    assert reg.size() == num_sample
    if num_sample != (0,):
        assert (orig_reg - reg).abs().max() < 1e-4


@pytest.mark.parametrize('num_sample', [
    (0,),
    (4,),
    (3, 4),
    (2, 5, 4)
])
def test_roi2reg_cross(num_sample):
    orig_roi = torch.rand(*num_sample)
    anchor = torch.rand(*num_sample)

    reg = roi2reg(orig_roi, anchor)
    roi = reg2roi(reg, anchor)

    assert roi.size() == num_sample
    if num_sample != (0,):
        assert (orig_roi - roi).abs().max() < 1e-4
