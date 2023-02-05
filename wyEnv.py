from keras.utils import np_utils
from keras.datasets import mnist
import random

class wyEnv():

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.x_train_r = self.x_train.reshape(*self.x_train.shape,1).astype('float32')
        self.x_test_r = self.x_test.reshape(*self.x_test.shape,1).astype('float32')

        self.x_train_n = self.x_train_r / 255.
        self.x_test_n = self.x_test_r / 255.

        self.y_train_oh = np_utils.to_categorical(self.y_train)
        self.y_test_oh = np_utils.to_categorical(self.y_test)

        self.action_space = len(set(self.y_test))-1
        self.state = self.sample_state()

    def reset(self):
        state,_ = self.step(-1)
        return state

    def step(self,action):
        if action == -1:
            current = self.state
            self.state = self.sample_state()
            return current,0
        r = self.reward(action)
        self.state = self.sample_state()
        return self.state, r

    def reward(self,action):
        y = self.y_train[self.state]
        if y == action:
            return 1

        else:
            return -1

    def sample_state(self):
        return random.randint(0,len(self.y_train)-1)

    def sample_action(self):
        return random.randint(0,self.action_space)
