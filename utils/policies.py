import numpy as np


def get_random_policy(actions_no: int) -> function:
    def random_policy(state: int) -> int:
        return np.random.randint(actions_no)
    return random_policy
