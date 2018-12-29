import numpy as np
import pytest

from examples import frozen_lake_iterative_policy_evaluation as policy_eval


@pytest.mark.parametrize(
    'gamma, expected_result',
    (
        (
            1,
            np.array([0.0] * 13 + [1.0, 1.0, 0.0])
        ),
        (
            0,
            np.array([0.0] * 14 + [1.0, 0.0])
        ),
    )
)
def test_find_value_function(gamma, expected_result):

    result = policy_eval.find_value_function(gamma=gamma)
    np.testing.assert_equal(result, expected_result)
