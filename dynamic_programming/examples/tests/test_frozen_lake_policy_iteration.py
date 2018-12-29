import numpy as np

from examples import frozen_lake_policy_iteration as policy_iter


def test_find_value_function():
    expected_result = [  # optimal policy
        {1, 2},
        {2},
        {1},
        {0},
        {1},
        {0, 1, 2, 3},
        {1},
        {0, 1, 2, 3},
        {2},
        {1, 2},
        {1},
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {2},
        {2},
        {0, 1, 2, 3},
    ]
    agent = policy_iter.find_optimal_policy_by_policy_iteration(iterations=20)
    result = []
    for state in range(16):
        state_actions = []
        for _ in range(50):
            state_actions.append(agent.policy(state))
        result.append(set(state_actions))

    np.testing.assert_equal(result, expected_result)
    assert result == expected_result
