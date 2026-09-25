import numpy as np
import pytest

from scripts.boundary_ink_overlap import auc, rank


def test_auc_ties_and_rank_ties() -> None:
    values = np.array([1., 2., 2., 3.])
    uncertain = np.array([True, True, False, False])
    assert auc(values, uncertain) == pytest.approx(.875)
    assert rank(values).tolist() == [0., 1.5, 1.5, 3.]
