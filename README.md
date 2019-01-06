#### Description
Goal of this project is to created simple environment for 
using Dynamic Programing methods in 
[OpenAI Gym](https://github.com/openai/gym) Frozen Lake environment.

DP methods required full knowledge about the MDP and because of 
that they couldn't be applied directly to Frozen Lake env.
 
Environment `custom_frozen_lake` allows to put agent in any 
state and by estimating rewards and transition probabilities
gives the possibility to apply DP.

Value Iteration and Policy Iteration methods were implemented 
based on slides from David Silver 
[Reinforcement Course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLweqsIcZJac7PfiyYMvYiHfOFPg9Um82B) at UCL.

#### Usage
You can run examples for applying simply by running 
`python run_example.py` in `dynamic_programming` directory.

Most objects has comments about usage so feel free to check them.

Have fun! 