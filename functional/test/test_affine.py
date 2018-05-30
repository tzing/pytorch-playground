import pytest
from pytest import mark
import torch

from functional import affine


@pytest.fixture
def mat():
    return torch.tensor([[ 1.16438634e+02, -3.86009325e+01,  3.86009325e+03],
       [ 1.00000000e+02,  1.00000000e+02,  0.00000000e+00],
       [ 8.53553391e-01,  1.46446609e-01,  5.05000000e+01]])


@mark.parametrize('pt_shape', [(0,), (2,), (1, 2), (10, 2), (3, 7, 2)])
def test_warp_points(pt_shape, mat):
    pt = torch.rand(*pt_shape) * 100

    pt = affine.warp_points(pt, mat)

    assert pt.size(-1) in (0, 2)


@mark.parametrize('pt_shape', [(0,), (2,), (1, 2), (10, 2), (3, 7, 2)])
def test_warp_points_inv(pt_shape, mat):
    pt = torch.rand(*pt_shape) * 100

    pt = affine.warp_points_inv(pt, mat)

    assert pt.size(-1) in (0, 2)


@mark.parametrize('pt_shape', [(0,), (2,), (1, 2), (10, 2), (3, 7, 2)])
def test_warp_points_cross(pt_shape, mat):
    pt = torch.rand(*pt_shape) * 100

    pt_t = affine.warp_points(pt, mat)
    pt_e = affine.warp_points_inv(pt_t, mat)

    assert pt_shape == (0,) or (pt - pt_e).abs().max() < 1e-4


@mark.parametrize('pt_shape', [(0,), (2,), (1, 2), (10, 2), (3, 7, 2)])
def test_warp_points_inv_cross(pt_shape, mat):
    pt = torch.rand(*pt_shape) * 100

    pt_t = affine.warp_points_inv(pt, mat)
    pt_e = affine.warp_points(pt_t, mat)

    assert pt_shape == (0,) or (pt - pt_e).abs().max() < 1e-4


@mark.parametrize('bx_shape', [(0,), (4,), (1, 4), (10, 4), (3, 7, 4)])
def test_warp_box(bx_shape, mat):
    bx = torch.rand(*bx_shape)
    if bx_shape != (0,):
        bx[..., :2] *= 100
        bx[..., 2:] *= 10

    pbx = affine.warp_box(bx, mat)
    assert pbx.size() == bx_shape


@mark.parametrize('bx_shape', [(0,), (4,), (1, 4), (10, 4), (3, 7, 4)])
def test_warp_box_inv(bx_shape, mat):
    bx = torch.rand(*bx_shape)
    if bx_shape != (0,):
        bx[..., :2] *= 100
        bx[..., 2:] *= 10

    pbx = affine.warp_box_inv(bx, mat)
    assert pbx.size() == bx_shape

