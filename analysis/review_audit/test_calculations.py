"""Check the mathematical controls against exhaustive enumeration."""

import itertools

import numpy as np
import pytest

from analysis.review_audit.run_audit import expected_max, paired_summary


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_subset_expectation_matches_enumeration(k):
    similarities = np.array([[0.1, 0.4, 0.2, 0.8, 0.5], [0.8, 0.3, 0.7, 0.1, 0.5]])
    enumerated = np.mean(
        [similarities[:, list(subset)].max(axis=1) for subset in itertools.combinations(range(5), k)], axis=0
    )
    np.testing.assert_allclose(expected_max(similarities, list(range(5)), k), enumerated)


def test_ineligible_reference_returns_missing():
    assert np.isnan(expected_max(np.ones((2, 2)), [0, 1], 3)).all()


def test_exact_sign_flip_known_case():
    result = paired_summary([0, 0], [1, 2])
    assert result["difference"] == 1.5
    assert result["exact_sign_flip_p"] == 0.5
