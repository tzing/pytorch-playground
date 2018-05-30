import torch
import math

import functional
from .exception import assert_last_dim


DEFAULT_METHOD='mean'


def warp_points(points, matrix):
    assert_last_dim(points, 2)
    if len(points) == 0:
        return points

    if len(points.size()) == 1:
        points = points.view(1, 2)

    o_points = torch.ones(*points.shape[:-1], 3, device=points.device)
    o_points[..., :2] = points

    t_points = o_points.matmul(matrix.t())
    t_points = t_points[..., :2] / t_points[..., [2]]

    return t_points


def warp_points_inv(points, matrix):
    assert_last_dim(points, 2)
    if len(points) == 0:
        return points

    if len(points.size()) == 1:
        points = points.view(1, 2)

    t_points = torch.ones(*points.shape[:-1], 3, device=points.device)
    t_points[..., :2] = points

    o_points = t_points.matmul(matrix.inverse().t())
    o_points = o_points[..., :2] / o_points[..., [2]]

    return o_points


def _internal_warp_box(box, matrix, warp_func, method):
    assert_last_dim(box, 4)
    if len(box) == 0:
        return box

    shape = box.shape[:-1]

    # center
    p_center = warp_func(box[..., :2], matrix)

    # corners
    lt_br = functional.center2corner_u(box)
    corner = lt_br[..., [0,1,2,3,0,3,2,1]].view(*shape, 4, 2) # magic!

    p_corner = warp_func(corner, matrix)
    dist = p_corner - p_center.unsqueeze(-2)

    # dist to corners
    if method == 'mean':
        p_size = dist.abs().mean(-2)

    elif method == 'min':
        p_size, _ = dist.abs().min(-2)

    else:
        raise ValueError('Unsupported method: ' + method)

    return torch.cat([p_center, p_size * 2], dim=-1).view(*shape, 4)


def warp_box(box, matrix, method=DEFAULT_METHOD):
    """
    Warp box using affine transform

    param
    -----
    - box: (torch.tensor) the boxes to project
    - matrix: (torch.tensor) the affine matrix

    return
    ------
    - box: (torch.tensor) projected boxes
    """
    return _internal_warp_box(box, matrix, warp_points, method)


def warp_box_inv(box, matrix, method=DEFAULT_METHOD):
    """
    Inverse warp box using affine transform

    param
    -----
    - box: (torch.tensor) the boxes to project
    - matrix: (torch.tensor) the affine matrix

    return
    ------
    - box: (torch.tensor) projected boxes
    """
    return _internal_warp_box(box, matrix, warp_points_inv, method)
