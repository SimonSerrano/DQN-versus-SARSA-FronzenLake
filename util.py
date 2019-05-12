import matplotlib.pyplot as plt
import seaborn as sns
import gym
import numpy as np

def game_avg(agent,env,e):
    rew_average = 0.
    for i in range(100):
        state = env.reset()
        done = False
        while done != True:
            action = agent.best_action(state)
            state, r, done, info = env.step(action)  # take step using selected action
            rew_average += r
    rew_average = rew_average / 100
    print('Episode {} avarage reward: {}'.format(e, rew_average))
    return rew_average


def cumm_plot(rewards):
    plt.subplot()
    cummulative = np.cumsum(rewards)
    plt.scatter([i + 1 for i in range(len(rewards))], cummulative)
    plt.show()


def scatter_plot(rewards):
    plt.subplot()
    plt.scatter([i + 1 for i in range(len(rewards))], rewards)
    plt.show()
