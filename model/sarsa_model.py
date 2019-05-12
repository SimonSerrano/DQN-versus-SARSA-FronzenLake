import numpy as np
from collections import deque
import random
import pickle

class Brain_SARSA:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = 1 #discount rate
        self.start_epsilon = 0.5#exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        model = np.ones(shape=(self.state_shape, self.action_size))

        # model = np.random.ra([self.state_shape,self.action_size])

        return model


    def act(self, state):
        #self.start_epsilon *= self.epsilon_decay
        #self.start_epsilon = max(self.epsilon_min, self.start_epsilon)
        if np.random.rand() < self.start_epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.model[state, :])
        #return max(list(range(self.action_size)), key=lambda x: self.model[(state, x)])

    def best_action(self,state):
        return np.argmax(self.model[state, :])



    def learn(self, state, action, reward, new_state, next_action):
            predict = self.model[state, action]
            target = reward + self.gamma * self.model[new_state,next_action] - predict
            self.model[state, action] = self.model[state, action] + self.learning_rate * target


    def load(self, name):
        with open(name, 'rb') as handle :
            self.model = pickle.load(handle)

    def save(self, name):
        with open(name, 'wb') as handle :
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)