import random
import numpy as np
from collections import deque
from keras.models import *
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
import os

batch_size = 32
output_dir = 'model_output/gridWorld'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('oi')

class DQNAgent:
    
    def __init__(self, action_size, image_size):
        self.action_size = action_size
        self.image_size = image_size
        
        self.memory = deque(maxlen=2000)
        
        self.discount_factor = 0.95

        self.greedy_value = 1.0
        self.greedy_value_decay = 0.995
        self.greedy_value_min = 0.01
        
        self.learning_rate = 0.001
        
        self.model = self._build_model()
        
    def _build_model(self):
        
        # Sequential Model
        model = Sequential()
		
		# 1st cnn layer
        model.add(Conv2D(32, kernel_size=8, strides=4, 
                 kernel_initializer="normal", 
                 padding="same",
                 input_shape=(64, 84, 1)))
        model.add(Activation("relu"))
		
        # 2st cnn layer
        model.add(Conv2D(64, kernel_size=4, strides=2, 
                 kernel_initializer="normal", 
                 padding="same"))
        model.add(Activation("relu"))
		
		# 3st cnn layer
        model.add(Conv2D(64, kernel_size=3, strides=1,
                 kernel_initializer="normal",
                 padding="same"))
        model.add(Activation("relu"))
		
		# Flattening parameters
        model.add(Flatten())
		
		# 1st mlp layer
        model.add(Dense(512, kernel_initializer="normal"))
        model.add(Activation("relu"))
		
		# 2st (last) cnn layer -> Classification layer (up, down, right, left)
        model.add(Dense(self.action_size, kernel_initializer="normal"))

		
		# Compiling Model
        model.compile(optimizer=Adam(lr=1e-6), loss="mse")

		# Show model details
        model.summary()
		
        return model
    
    def remember(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        
        if np.random.rand() <= self.greedy_value:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        
        mini_batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in mini_batch:
            
            target = reward
            
            if not done:
                target = (reward + self.discount_factor * np.argmax(self.model.predict(next_state)[0]))
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.greedy_value > self.greedy_value_min:
            self.greedy_value *= self.greedy_value_decay
            
    def load(self, name):
        self.model.load_weights(name)
     
    def save(self, name):
        self.model.save_weights(name)
                