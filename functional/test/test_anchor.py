import pytest
import numpy

from functional.anchor import make_anchor


@pytest.mark.parametrize("n_in, n_feat, n_aspect, base_size", [
    (224, 1, 1, 8),
    (384, 7, 3, 16),
])
def test_make_anchor(n_in, n_feat, n_aspect, base_size):
    aspect = numpy.random.rand(n_aspect, 2)
    anchor = make_anchor([n_in] * 2, [n_feat] * 2, base_size, aspect)
    assert anchor.size() == (n_feat, n_feat, n_aspect, 4)
