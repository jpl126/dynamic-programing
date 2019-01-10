import pytest
from environments.frozen_lake import custom_frozen_lake
from utils.errors import InvalidStateError, InvalidMoveError


@pytest.fixture
def env():
    return custom_frozen_lake.FrozenLakeEnv()


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            [0, 0],
            {0, 3}
        ),
        (
            [3, 3],
            {1, 2}
        ),
        (
            [3, 1],
            {1}
        ),
        (
            [2, 2],
            set()
        )
    )
)
def test_get_walls_next_to_agent(env, example, expected_result):
    env._agent_position = example
    result = env._get_walls_next_to_agent()
    assert result == expected_result


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            (1, 1),
            5
        ),
        (
            (3, 3),
            15
        ),
    )
)
def test_get_goal_observation(example, expected_result):
    env = custom_frozen_lake.FrozenLakeEnv(goal_position=example)
    result = env._get_goal_observation()
    assert result == expected_result


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            2,
            [0, 1]
        ),
        (
            1,
            [1, 0]
        )
    )
)
def test_change_agent_position(env, example, expected_result):
    env._change_agent_position(example)
    assert env._agent_position == expected_result


@pytest.mark.parametrize(
    'example',
    (
        4,
        5
    )
)
def test_change_agent_position_error(env, example):
    with pytest.raises(InvalidMoveError):
        env._change_agent_position(example)


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            (1, 1),
            [1, 1],
        ),
        (
            (3, 3),
            [3, 3],
        )
    )
)
def test_change_agent_position_in_hole_or_at_goal(example, expected_result):
    env = custom_frozen_lake.FrozenLakeEnv(start_position=example)
    for i in range(4):
        env._change_agent_position(i)
        assert env._agent_position == expected_result


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            [0, 0],
            0
        ),
        (
            [3, 3],
            15
        ),
        (
            [3, 1],
            13
        ),
        (
            [2, 2],
            10
        )
    )
)
def test_get_observation(env, example, expected_result):
    env._agent_position = example
    result = env._get_observation()
    assert result == expected_result
    assert result == env.observation


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            4,
            [1, 0]
        ),
        (
            1,
            [0, 1]
        )
    )
)
def test_set_observation(env, example, expected_result):
    env.observation = example
    assert env._agent_position == expected_result


@pytest.mark.parametrize(
    'example',
    (
        16,
        204
    )
)
def test_set_observation_error(env, example):
    with pytest.raises(InvalidStateError):
        env.observation = example


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            2,
            2
        ),
        (
            1,
            1
        )
    )
)
def test_observation(env, example, expected_result):
    env.observation = example
    assert env.observation == expected_result


@pytest.mark.parametrize(
    'example, initial_position, expected_result',
    (
        (
            2,
            [0, 0],
            (1, 0, False, {})
        ),
        (
            1,
            [0, 0],
            (4, 0, False, {})
        ),
        (
            0,
            [0, 0],
            (0, 0, False, {})
        ),
        (
            2,
            [3, 2],
            (15, 1, True, {})
        ),
        (
            1,
            [0, 1],
            (5, 0, True, {})
        ),
        (
            1,
            [3, 3],
            (15, 0, True, {})
        ),
    )
)
def test_step(env, example, initial_position, expected_result):
    env._agent_position = initial_position
    result = env.step(example)
    assert result == expected_result


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            [1, 1],
            True
        ),
        (
            [2, 0],
            False
        ),
    )
)
def test_is_agent_in_hole(env, example, expected_result):
    env._agent_position = example
    result = env._is_agent_in_hole()
    assert result == expected_result


def test_states_count(env):
    assert env.states_count == 16


def test_action_count(env):
    assert env.actions_count == 4


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            (3, 3),
            True
        ),
        (
            (1, 1),
            False
        ),
    )
)
def test_is_agent_at_goal(example, expected_result):
    env = custom_frozen_lake.FrozenLakeEnv(start_position=example)
    assert env._is_agent_at_goal() == expected_result


@pytest.mark.parametrize(
    'example, expected_result',
    (
        (
            (3, 3),
            15,
        ),
        (
            (1, 1),
            5,
        ),
    )
)
def test_reset(example, expected_result):
    env = custom_frozen_lake.FrozenLakeEnv(start_position=example)
    env.reset()
    assert env.observation == expected_result
