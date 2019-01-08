from examples import frozen_lake_iterative_policy_evaluation
from examples import frozen_lake_policy_iteration
from examples import frozen_lake_value_iteration


if __name__ == '__main__':
    print('Choose example:\n1 - Iterative Policy Evaluation\n'
          '2 - Policy Iteration\n3 - Value Iteration')
    examples = {
        '1': frozen_lake_iterative_policy_evaluation,
        '2': frozen_lake_policy_iteration,
        '3': frozen_lake_value_iteration
    }
    decision = input('Your choice: ')

    if decision in examples:
        examples[decision].main()
    else:
        print('Wrong example. Quiting...')

