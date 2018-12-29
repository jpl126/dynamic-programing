from examples import frozen_lake_policy_iteration
from examples import frozen_lake_iterative_policy_evaluation


if __name__ == '__main__':
    print('Choose example:\n1 - Iterative Policy Evaluation\n'
          '2 - Policy Iteration')
    examples = {
        1: frozen_lake_iterative_policy_evaluation,
        2: frozen_lake_policy_iteration,
    }
    decision = int(input('Your choice: '))

    if decision in set(examples):
        examples[decision].main()
    else:
        print('Wrong example. Quiting...')

