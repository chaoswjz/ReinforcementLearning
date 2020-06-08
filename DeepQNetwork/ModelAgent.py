#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, size):
        self._size = size
        self._memory = []

    def __len__(self):
        return len(self._memory)

    def getBatch(self):
        return random.sample(self._memory, batch_size)

    def store(self, transition):
        self._memory.append(transition)
        if len(self._memory) > self._size:
            self._memory.pop(0)


class DQN(Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()

        self.fc1 = Dense(1024)
        self.act1 = Activation('relu')
        self.fc2 = Dense(512)
        self.act2 = Activation('relu')
        self.fc3 = Dense(action_size)
        self.act3 = Activation('softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)

        return x


class Agent:
    def __init__(
            self,
            features,
            actions,
            learning_rate=0.01,
            discount=0.9,
            e_greedy=0.1,
            batch_size=32,
            capacity=1000,
    ):
        # experience replay
        self._memory = ReplayBuffer(capacity)
        self._model = DQN(actions)
        # fixed q target
        self._target = DQN(actions)
        self._memory_size = capacity
        self._actions = actions
        self._features = features
        self._lr = learning_rate
        self._discount = discount
        self._e_greedy = e_greedy
        self._batch_size = batch_size

    def takeAction(self, state):
        prob = np.random.uniform()
        if prob < self._e_greedy:
            return np.random.randint(0, self._actions)
        else:
            state = tf.Variable(state, dtype=tf.float)
            state_action = self._model(state)
            return np.argmax(state_action)

    def updateTarget(self):
        self._target.set_weights(self._model.get_weights())
