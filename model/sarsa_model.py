import numpy as np
from collections import deque
import random
import pickle

class Brain_SARSA:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 #discount rate
        self.epsilon = 1.0 #exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.099
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        model = np.zeros(shape=(self.state_shape, self.action_size))
        return model

    def remember(self,state,action,reward,next_state, next_action,done):
        self.memory.append((state, action, reward, next_state, next_action, done))

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model[state, :])


    def learn(self, batch_size):
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, next_action, _ = sample
            Q_future = self.model[new_state, next_action]
            target = reward + Q_future * self.gamma
            self.model[state, action] = self.model[state, action] + self.learning_rate * (target - Q_future)


    def load(self, name):
        with open(name, 'rb') as handle :
            self.model = pickle.load(handle)

    def save(self, name):
        with open(name, 'wb') as handle :
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)