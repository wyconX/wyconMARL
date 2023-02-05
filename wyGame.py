import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.experimental.output_all_intermediates(True)

from keras.utils import np_utils
from keras.models import Model
import os.path
import numpy as np
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from collections import deque
import random,time

from wyPolicy import wyPolicy
from wyEnv import wyEnv
from wyAgent import wyAgent

class wyGame():
    def __init__(self):
        self.env = wyEnv()
        self.action_num = len(set(self.env.y_test))
        self.dummy_actions = np.ones((1, self.action_num))
        self.state = self.env.x_train_r[self.env.state]
        _, self.input_h, self.input_w = self.env.x_test.shape

        self.policy = wyPolicy(esp_total=2020)

        self.actor = wyAgent(self.action_num)
        self.actor.build_model(self.input_h, self.input_w)
        self.actor.setEnv(self.env)
        self.actor.setPolicy(self.policy)

        self.critic = wyAgent(self.action_num)
        self.critic.build_model(self.input_h, self.input_w)
        self.critic.setEnv(self.env)
        self.critic.setPolicy(self.policy)
        self.q = Model(inputs=self.actor.model.input, outputs=self.actor.model.get_layer('output_q').output)

        self.memory = deque(maxlen=512)
        self.reward_rec = []
        self.total_rewards = 0
        self.update_step = 128
        self.forward = 512
        self.stage = 0
        self.iteration = 20
        self.hit = 0

    def copy_c_to_a(self):
        c_weights = self.critic.model.get_weights()
        a_weights = self.actor.model.get_weights()
        for i in range(len(c_weights)):
            a_weights[i] = c_weights[i]
        self.actor.model.set_weights(a_weights)

    def explore(self, num):
        s = self.env.reset()
        self.state = self.env.x_train_n[s]
        for i in range(num):
            rand_action = self.env.sample_action()
            next_state, reward = self.env.step(rand_action)
            self.remember(self.state, rand_action, 0, reward, next_state)
            self.state = self.env.x_train_n[next_state]

    def remember(self, state, action, q_value, reward, next_state):
        self.memory.append([state, action, q_value, reward, next_state])

    def rand_sample(self, num):
        return np.array(random.sample(self.memory, num))

    def get_q_value(self, state, action):
        input = [state.reshape(1,*state.shape), action]
        q_values = self.q.predict(input)
        return q_values

    def select_action(self, state, step, dummy):
        eps = self.policy.epsilon(step)
        if np.random.rand() < eps:
            return self.env.sample_action(), 0
        self.hit+=1
        q_values = self.get_q_value(state, dummy)
        return np.argmax(q_values), np.max(q_values)

    def writeRewards(self,reward):
        path = "wyconReward.txt"
        path = os.path.join(os.getcwd(), path)
        reward = "%s" % (reward)

        with open(path, "a") as f:
            f.writelines(reward)
            f.writelines('\n')

    def critic_learn(self, sample_size):
        '''filepath = "wyCritic.h5"

        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor='val_accuracy',
            verbose=0,
            save_weights_only=True,
            period=1
        )'''

        if len(self.memory) < sample_size:
            print('memory not enough')
            return

        samples = self.rand_sample(sample_size)

        states, actions, old_q, rewards, next_states = zip(*samples)
        states, actions, old_q, rewards = np.array(states), np.array(actions).reshape(-1, 1), np.array(old_q).reshape(-1, 1), np.array(rewards).reshape(-1, 1)

        action_oh = np_utils.to_categorical(actions, self.action_num)

        q = 0
        q_estimate = (1 - self.critic.alpha) * old_q + self.critic.alpha * (rewards.reshape(-1,1) + self.critic.gamma * q)
        history = self.critic.model.fit([states, action_oh], q_estimate, epochs=1, verbose=0)

        return np.mean(history.history['loss'])

    def train(self, epochs):
        filepath = "wyCritic.h5"
        if os.path.exists(filepath):
            self.critic.model.load_weights(filepath)
            self.copy_c_to_a()
            print('load success')
        pbar = tqdm(range(1, epochs + 1))

        for epoch in pbar:
            start = time.time()
            self.total_rewards = 0
            for step in range(self.forward):
                action, q = self.select_action(self.state, self.stage, self.dummy_actions)
                eps = self.policy.epsilon(self.stage)
                
                next_state, reward = self.env.step(action)
                self.remember(self.state, action, q, reward, next_state)

                loss = self.critic_learn(64)
                self.total_rewards += reward
                self.state = self.env.x_train_n[next_state]

                if step % self.update_step == 0:
                    self.copy_c_to_a()

            self.stage += 1
            self.reward_rec.append(self.total_rewards)
            self.writeRewards(self.total_rewards)
            pbar.set_description('R:{} L:{:.4f} T:{} P:{:.3f} H:{}'.format(self.total_rewards, loss, int(time.time() - start), eps, self.hit))
            self.actor.count = 0

        self.critic.model.save_weights(filepath)
        print('saved')
        '''K.clear_session()
        self.critic = wyAgent(self.action_num)
        self.critic.build_model(self.input_h, self.input_w)
        self.critic.setEnv(self.env)
        self.critic.setPolicy(self.policy)
        self.critic.model.load_weights(filepath)
        self.actor = wyAgent(self.action_num)
        self.actor.build_model(self.input_h, self.input_w)
        self.actor.setEnv(self.env)
        self.actor.setPolicy(self.policy)'''

    def test(self):
        filename = 'wyCritic.h5'
        if os.path.isfile(filename):
            self.actor.model.load_weights(filename)
        else:
            return
        inputs = [self.env.x_test_n, np.ones(shape=(len(self.env.y_test), self.action_num))]
        q_values = self.q.predict(inputs)
        pred = np.argmax(q_values,axis=1)

        accuracy = accuracy_score(self.env.y_test,pred)
        print(accuracy)


    def play(self, epochs=1000, is_train=True):
        filename = "wyCritic.h5"
        self.memory.clear()
        repeat = epochs//self.iteration
        remainder = epochs%self.iteration
        s = self.env.reset()
        self.state = self.env.x_train_n[s]

        if is_train:
            self.explore(256)
            for i in range(repeat):
                self.train(self.iteration)
            self.train(remainder)
            self.critic.model.save_weights(filename)
            print('saved')
            self.test()
        else:
            self.test()



if __name__=='__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    graph = tf.Graph().finalize()
    sess = tf.Session(config=config,graph=graph)
    tf.keras.backend.set_session(sess)

    g = wyGame()
    #print(g.critic.model.summary())
    #g.policy = wyPolicy(esp_total=1)

    g.play(2000)
    #g.test()





