import pytest
import torch

from functional import coord


@pytest.fixture
def input1d():
    return torch.rand(4)


@pytest.fixture
def input2d():
    return torch.rand(3, 4)


@pytest.fixture
def input3d():
    return torch.rand(3, 5, 4)


# _split_input_I ---------------------------------------------------------------
@pytest.mark.parametrize('input_', [input1d, input2d, input3d])
def test__split_input_I(input_):
    input_ = input_()
    p0, p1 = coord._split_input([input_])
    assert p0.size() == p1.size()


@pytest.mark.parametrize('input_', [input1d, input2d, input3d])
def test__split_input_II(input_):
    input_ = input_()
    p0, p1 = coord._split_input([input_[..., :2], input_[..., 2:]])
    assert p0.size() == p1.size()


# center2corner ----------------------------------------------------------------
@pytest.mark.parametrize('input_', [input1d, input2d, input3d])
def test_center2corner(input_):
    input_ = input_()
    lt, br = coord.center2corner(input_)
    assert lt.size() == (*input_.shape[:-1], 2)
    assert br.size() == (*input_.shape[:-1], 2)


@pytest.mark.parametrize('input_', [input1d, input2d, input3d])
def test_center2corner_u(input_):
    input_ = input_()
    output = coord.center2corner_u(input_)
    assert output.size() == input_.size()


# corner2center ----------------------------------------------------------------
@pytest.mark.parametrize('input_', [input1d, input2d, input3d])
def test_corner2center(input_):
    input_ = input_()
    xy, wh = coord.corner2center(input_)

    assert xy.size() == (*input_.shape[:-1], 2)
    assert wh.size() == (*input_.shape[:-1], 2)


@pytest.mark.parametrize('input_', [input1d, input2d, input3d])
def test_corner2center_u(input_):
    input_ = input_()
    output = coord.corner2center_u(input_)
    assert output.size() == input_.size()


# center2corner cross ----------------------------------------------------------
@pytest.mark.parametrize('input_', [input1d, input2d, input3d])
def test_center2corner_cross(input_):
    input_ = input_()
    p0, p1 = input_[..., :2], input_[..., 2:]

    lt, br = coord.center2corner(p0, p1)
    xy, wh = coord.corner2center(lt, br)
    assert (xy - p0).abs().max() < 1e-4
    assert (wh - p1).abs().max() < 1e-4


# corner2center cross ----------------------------------------------------------
@pytest.mark.parametrize('input_', [input1d, input2d, input3d])
def test_corner2center_cross(input_):
    input_ = input_()
    xy, wh = input_[..., :2], input_[..., 2:]
    xy *= 10
    lt = xy - wh
    br = xy + wh

    xy, wh = coord.corner2center(lt, br)
    flt, fbr = coord.center2corner(xy, wh)
    assert (lt - flt).abs().max() < 1e-4
    assert (br - fbr).abs().max() < 1e-4
