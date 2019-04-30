from keras.layers import Input, Dense, Flatten, Conv1D
from keras.models import Model
from keras import initializers, optimizers
import numpy as np
from collections import deque
import random

class Brain_DQN:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 #discount rate
        self.epsilon = 1.0 #exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 1.0
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        NB_NEURONS = 24
        #Define input
        inputs =Input(shape=self.state_shape)
        #Define hidden layers
        x = Dense(NB_NEURONS, activation='relu')(inputs)

        #Define output layer
        predictions = Dense(self.action_size, activation='softmax')(x)

        # Define a traning model
        model = Model(input=inputs, output=predictions)


        # Compile the model
        model.compile(loss='mse',
                      optimizer=optimizers.RMSprop(lr=self.learning_rate),
                      metrics=['accuracy'])

        # Print a summary of the model
        model.summary()

        return model

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

#Adapt to SARSA method
    def learn(self, batch_size):
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = reward
            if not done:
                Q_future = max(self.model.predict(new_state)[0])
                target = reward + Q_future * self.gamma
            target_f = self.model.predict(state)
            target_f[0, action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)