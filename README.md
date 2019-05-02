# DQN-versus-SARSA-FronzenLake
Study to compare the DQN (Q-learning) and SARSA learning  on the frozen lake game using gym (python)

# Expected
- Try bigger game than the 4 * 4
- What did you want to study ?
- What did you get as result?

# Usage
usage: program.py [-h] [--agent AGENT] [--episodes EPISODES] [--weights PATH]
                  [--batch BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --agent AGENT, -a AGENT
                        Specify to configure the agent to play the game : dqn,
                        sarsa
  --episodes EPISODES, -e EPISODES
                        Specify to configure the number of episodes, default
                        is 200
  --weights PATH, -w PATH
                        Specify to configure the path of the weights. If not
                        specified, the weights are not loaded neither saved
  --batch BATCH_SIZE, -b BATCH_SIZE
                        Specify this to configure the size of the batch when
                        learning, default is 200



