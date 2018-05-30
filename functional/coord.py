import torch

from .exception import assert_last_dim


def center2corner(*args) -> (torch.Tensor, torch.Tensor):
    """
    Convert the bounding box presentation from [center, size] to [left-top xy,
    bottom-right xy]

    param:
    - box: (torch.tensor) n by 4 array of the boxes

    return:
    - lt: (torch.tensor) the left-top location
    - br: (torch.tensor) the bottom-right location
    """
    xy, wh = _split_input(args)
    wh = wh / 2

    lt = xy - wh
    br = xy + wh

    return lt, br


def center2corner_u(*args) -> torch.Tensor:
    """
    Convert the bounding box presentation from [center, size] to [left-top xy,
    bottom-right xy]

    param:
    - box: (torch.tensor) n by 4 array of the boxes

    return:
    - box: (torch.tensor) converted boxes
    """
    return torch.cat(center2corner(*args), dim=-1)


def corner2center(*args) -> (torch.Tensor, torch.Tensor):
    """
    Convert the bounding box presentation from [left-top xy, bottom-right xy]
    to [center, size]

    param:
    - box: (torch.tensor) n by 4 array of the boxes in corner representation

    return:
    - xy: (torch.tensor) the center location
    - wh: (torch.tensor) the box sizes
    """
    lt, br = _split_input(args)

    wh = torch.abs(br - lt)
    xy = torch.min(lt, br) + wh / 2

    return xy, wh


def corner2center_u(*args) -> torch.Tensor:
    """
    Convert the bounding box presentation from [left-top xy, bottom-right xy]
    to [center, size]

    param:
    - box: (torch.tensor) n by 4 array of the boxes in corner representation

    return:
    - box: (torch.tensor) converted boxes
    """
    return torch.cat(corner2center(*args), dim=-1)


def _split_input(args):
    """
    INTERNAL FUNCTION
    split the box tensor and support the described 2 form
    """
    if len(args) == 1:  # form I, n by 4 boxes
        box = args[0]

        assert_last_dim(box, 4)

        part0 = box[..., :2]
        part1 = box[..., 2:]

    elif len(args) == 2:  # form II, (lt, br) or (wh, sz)
        part0, part1 = args

        assert_last_dim(part0, 2)
        assert_last_dim(part1, 2)

    else:
        raise ValueError()

    return part0, part1
