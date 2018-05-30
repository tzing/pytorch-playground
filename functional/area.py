import torch

from functional.exception import assert_last_dim


def area_triangle(base_point, end_point):
    """Calc areas of each triangles.
    """
    assert_last_dim(base_point, 2)
    assert_last_dim(end_point, 2)
    assert base_point.device == end_point.device

    #
    base_point = base_point.view(-1, 1, 2)
    end_point = end_point.view(1, -1, 2)
    n, m, d = base_point.size(0), end_point.size(1), base_point.device

    direct = end_point - base_point
    direct = torch.cat([direct, torch.ones((n, m, 1), device=d)], dim=-1)

    #
    area = torch.empty((base_point.size(0), end_point.size(1)-1), device=base_point)
    for i, row in enumerate(direct):
        for j in range(len(row)-1):
            area[i, j] = torch.norm(torch.cross(row[j], row[j+1]))
