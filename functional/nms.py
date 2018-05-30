import torch

from .iou import iou
from .exception import assert_last_dim


def nms(box, score, threshold=.5) -> (torch.Tensor, torch.Tensor):
    """
    Non maximum suppression

    param:
    - box: (torch.tensor) [n, 4] array of bounding boxes; `xywh` for each row
    - score: (torch.tensor) [n, 1] array of box scores
    - threshold: (float) overlap threshold

    return:
    - box: (torch.tensor) [m, 4] array of merged bounding boxes
    - score: (torch.tensor) [m, 1] array of box scores
    - index: (torch.tensor) [m, 1] array indicates the cluster each box merged into
    - num: (torch.tensor) [m, 1] array; number of the box merge into this box
    """
    assert_last_dim(box, 4)

    # no need to calc
    if len(box) == 0:
        device = box.device
        return (
            torch.tensor([], device=device), # box
            torch.tensor([], device=device), # score
            torch.tensor([], dtype=torch.int64, device=device), # index
            torch.tensor([], dtype=torch.int64, device=device), # num
        )

    # reshape & cast
    box = box.view(-1, 4)
    score = score.view(-1)

    assert box.device == score.device
    assert box.size(0) == score.size(0)

    # eps
    threshold -= 1e-4

    # sort scores
    _, order = score.sort()
    _, rev_order = order.sort()

    is_avaliable = (box[:, 2] > 0) & (box[:, 3] > 0)
    index = torch.empty_like(score, dtype=torch.int64).fill_(-1)

    # pick
    pick = []
    num = []
    while is_avaliable.any():
        idx = order[is_avaliable][-1].item()
        score = iou(box[idx], box).view(-1)

        should_merge = score > threshold
        if not should_merge.any(): # corner case
            is_avaliable[rev_order[idx]] = 0
            continue

        pick.append(idx)
        idx_merge = rev_order[should_merge]
        is_avaliable[idx_merge] = 0
        index[idx_merge] = len(pick) - 1
        num.append(should_merge.long().sum().item())

    # return
    pick = torch.tensor(pick, dtype=torch.int64, device=box.device)
    num = torch.tensor(num, device=box.device)
    box = box[pick]
    score = score[pick]

    return box, score, index, num
