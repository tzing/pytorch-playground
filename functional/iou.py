import torch

from .coord import center2corner
from .exception import assert_last_dim


def iou(base, target) -> torch.Tensor:
    """
    Calculating Intersection-over-Union ratio

    param:
    - base: (torch.Tensor) n by 4 array of the based boxes
    - target: (torch.Tensor) m by 4 array of the targeting boxes

    return:
    - score: (torch.Tensor) n by m matrix of the score map
    """
    if len(base) == 0 or len(target) == 0:
        return torch.tensor([])

    assert_last_dim(base, 4)
    assert_last_dim(target, 4)

    # align dimensions
    base = base.view(-1, 1, 4)
    target = target.view(1, -1, 4)

    base_lt, base_br = center2corner(base)
    target_lt, target_br = center2corner(target)

    # area
    base_area = base[..., 2:].prod(-1)
    target_area = target[..., 2:].prod(-1)

    # calc
    lt = torch.max(base_lt, target_lt)
    br = torch.min(base_br, target_br)

    intersect = torch.clamp(br - lt, min=0).prod(dim=2)
    union = base_area + target_area - intersect
    iou = intersect / union
    return iou
