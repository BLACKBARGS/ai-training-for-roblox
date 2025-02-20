from tensorflow import keras
from tensorflow import losses, metrics
import numpy as np
import tensorflow as tf
from keras import layers, Model
from collections import deque
import random

class DQNetwork(Model):
    def __init__(self, action_space):
        super(DQNetwork, self).__init__()
        self.conv1 = layers.Conv2D(32, (8, 8), strides=4, activation='relu')
        self.conv2 = layers.Conv2D(64, (4, 4), strides=2, activation='relu')
        self.conv3 = layers.Conv2D(64, (3, 3), strides=1, activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(action_space)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.tau = 0.001
        self.model = DQNetwork(action_space)
        self.target_model = DQNetwork(action_space)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.build(input_shape=(None,) + state_shape)
        self.target_model.build(input_shape=(None,) + state_shape)
        self.update_target_network()

    def update_target_network(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)
        
        state = np.expand_dims(state, axis=0)
        q_values = self.model(state)
        return np.argmax(q_values[0])

    @tf.function
    def _compute_loss(self, states, actions, rewards, next_states, dones):
        future_rewards = self.target_model(next_states)
        max_future_rewards = tf.reduce_max(future_rewards, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_future_rewards
        
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            one_hot_actions = tf.one_hot(actions, self.action_space)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        
        return loss, tape

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        loss, tape = self._compute_loss(states, actions, rewards, next_states, dones)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.update_target_network()
        
        return loss.numpy()

    def save(self, path):
        self.model.save_weights(path + "_main")
        self.target_model.save_weights(path + "_target")

    def load(self, path):
        self.model.load_weights(path + "_main")
        self.target_model.load_weights(path + "_target")
