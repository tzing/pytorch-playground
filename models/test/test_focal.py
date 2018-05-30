import pytest
import torch

from models.focal import FocalLoss


@pytest.mark.parametrize('num_class', [2, 16])
@pytest.mark.parametrize('num_item', [1, 16])
def test_focal(num_item, num_class):
    data = torch.rand(num_item, num_class)
    label = torch.randint(num_class, size=(num_item,)).long()

    loss = FocalLoss()

    loss(data, label)
