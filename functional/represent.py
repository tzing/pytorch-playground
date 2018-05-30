import torch
import numpy

from .coord import center2corner
from .exception import assert_last_dim


def reg2roi(regress, anchor) -> (torch.Tensor, torch.Tensor):
    """
    Convert the target representation from the regression target to the world
    coordinate Region of Interest location.

    param:
    - regress: (torch.Tensor) the location in target representation
    - anchor: (torch.Tensor) the based anchor

    return:
    - roi: (torch.Tensor) the array in [ctr_x, ctr_y, w, h] representation of RoI

    NOTE:
    - this function support batch processing
    """
    if len(regress) == 0:
        return torch.tensor([])

    assert_last_dim(regress, 4)
    assert_last_dim(anchor, 4)

    # align dimension
    num_dim = numpy.maximum(anchor.size(), regress.size())
    num_dim = tuple(map(int, num_dim[:-1]))

    acr_xy = anchor[..., :2].expand(*num_dim, 2)
    acr_wh = anchor[..., 2:].expand(*num_dim, 2)

    # get roi location
    reg_xy = regress[..., :2]
    reg_wh = regress[..., 2:]

    roi_xy = acr_xy + reg_xy * acr_wh
    roi_wh = torch.exp(reg_wh) * acr_wh
    roi_wh[..., 2:].clamp_(min=0)

    roi = torch.cat([roi_xy, roi_wh], dim=-1)
    return roi


def roi2reg(roi, anchor) -> torch.Tensor:
    """
    Convert the RoI location from world coord to the target representation.

    param:
    - roi: (torch.Tensor) the RoI location in world coord
    - anchor: (torch.Tensor) the based anchor

    return:
    - reg: (torch.Tensor) the location in target representation
    """
    if len(roi) == 0:
        return torch.tensor([])

    assert_last_dim(roi, 4)
    assert_last_dim(anchor, 4)
    assert roi.size() == anchor.size()

    # get reg location
    roi_xy = roi[..., :2]
    roi_wh = roi[..., 2:].clamp(min=1e-4)

    acr_xy = anchor[..., :2]
    acr_wh = anchor[..., 2:]

    reg_xy = (roi_xy - acr_xy) / acr_wh
    reg_wh = torch.log(roi_wh / acr_wh)

    reg = torch.cat([reg_xy, reg_wh], dim=-1)
    return reg
