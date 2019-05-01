import gym
import numpy as np
import os
import argparse




def play_dqn(agent, env, args):
    wins = 0
    for e in range(args.episodes[0]):
        print("Starting episode : ", e)
        state = np.asarray([env.reset()])
        for _ in range(10000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.asarray([next_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done : 
                if reward == 1 : wins +=1
                agent.learn(args.batch_size[0])
                break
    print("Games won :", wins, "on", args.episodes[0])


def play_sarsa(agent, env, args):
    wins = 0
    for e in range(args.episodes[0]):
        print("Starting episode : ", e)
        state = env.reset()
        action = agent.act(state)
        for _ in range(10000):
            env.render()
            next_state, reward, done, _ = env.step(action)
            next_action = agent.act(next_state)
            agent.remember(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            if done : 
                if reward == 1 : wins +=1
                agent.learn(args.batch_size[0])
                break
    print("Games won :", wins, "on", args.episodes[0])

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", "-a", nargs=1, type=str, default="dqn", 
    help="Specify to configure the agent to play the game : dqn, sarsa")
    parser.add_argument("--episodes", "-e", dest="episodes", nargs=1, type= int, default=200, 
    help="Specify to configure the number of episodes, default is 200")
    parser.add_argument("--weights", "-w", dest="path", nargs=1, type=str, default=argparse.SUPPRESS,
    help="Specify to configure the path of the weights. If not specified, the weights are not loaded neither saved")
    parser.add_argument("--batch", "-b", dest="batch_size", nargs=1, type=int, 
    help="Specify this to configure the size of the batch when learning, default is 200")
    args = parser.parse_args()

    from model import dqn_model, sarsa_model

    env = gym.make("FrozenLake-v0", map_name='8x8')
    #state_space = env.observation_space -> Discrete(64)
    #action_space = env.action_space -> Discrete(4)
    agent = dqn_model.Brain_DQN((1,), 4)
    if args.agent[0] == "sarsa": 
        agent = sarsa_model.Brain_SARSA(64, 4)
    if args.path[0] and os.path.isfile(args.path[0]):
        agent.load(args.path[0])
    
    
    if args.agent[0] == "dqn":
        play_dqn(agent, env, args)
    if args.agent[0] == "sarsa" :
        play_sarsa(agent, env, args)

    if args.path[0]:
        agent.save(args.path[0])



if __name__ == "__main__":
    main()