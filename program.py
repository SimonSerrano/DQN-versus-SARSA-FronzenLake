import gym
from model import dqn_model as model
import numpy as np
import os

episodes = 200
batch_size = 200
PATH_DQN = './weights/dqn_agent'



def main() :
    env = gym.make("FrozenLake-v0", map_name='8x8')
    #state_space = env.observation_space -> Discrete(64)
    #action_space = env.action_space -> Discrete(4)
    agent = model.Brain_DQN((1,), 4)
    if(os.path.isfile(PATH_DQN)):
        agent.load(PATH_DQN)
    wins = 0
    for e in range(episodes):
        print("Starting episode : ", e)
        state = np.asarray([env.reset()])
        for t in range(10000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.asarray([next_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done : 
                if reward == 1 : wins +=1
                agent.learn(batch_size)
                break
    print("Games won :", wins, "on", episodes)
    agent.save(PATH_DQN)



if __name__ == "__main__":
    main()