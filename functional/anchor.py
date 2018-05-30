import torch
import numpy


def make_anchor(image_shape, feat_shape, base_size, aspects) -> torch.Tensor:
    """
    Generate anchor

    param:
    - image_shape: (list of 2 int) the size of the input image
    - feat_shape: (list of 2 int) the size of the feature map
    - base_size: (float) base size of the anchor
    - aspects: (list) the aspect ratio of the anchor

    return:
    - anchor: (torch.tensor) the anchors
    """
    iW, iH = image_shape
    fW, fH = feat_shape
    num_aspect = len(aspects)

    # base xy
    xx = numpy.linspace(0, iW, fW + 2)[1:-1]
    yy = numpy.linspace(0, iH, fH + 2)[1:-1]

    xx, yy = numpy.meshgrid(xx, yy)

    xx = torch.from_numpy(xx).view(fH, fW)
    yy = torch.from_numpy(yy).view(fH, fW)
    xy = (torch
          .stack([xx, yy], dim=-1)
          .float()
          .view(fH, fW, 1, 2)
          .expand(fH, fW, num_aspect, 2))

    # base wh
    area = base_size ** 2

    sizes = []
    for w, h in aspects:
        unit = numpy.sqrt(area / w / h)
        sizes.append((w * unit, h * unit))

    wh = (torch.tensor(sizes)
          .view(1, 1, num_aspect, 2)
          .expand(fH, fW, num_aspect, 2))

    anchor = torch.cat([xy, wh], dim=-1)
    return anchor
