import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import gym
import pylab
import numpy as np
from Environment import Env
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from Options import *

# 정책 신경망과 가치 신경망 생성
class A2C(tf.keras.Model):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.policy_fc1 = Dense(128, activation='tanh')
        self.drop1 = Dropout(0.2)
        self.policy_fc2 = Dense(256, activation='leaky_relu')
        self.drop2 = Dropout(0.2)
        self.policy_fc3 = Dense(256, activation='leaky_relu')
        self.drop3 = Dropout(0.2)
        self.policy_fc4 = Dense(128, activation='leaky_relu')
        self.actor_out = Dense(action_size, activation='softmax',
                               kernel_initializer=RandomUniform(-1e-3, 1e-3))

        self.critic_fc1 = Dense(128, activation='selu')
        self.critic_fc2 = Dense(64, activation='leaky_relu')
        self.critic_out = Dense(1, kernel_initializer =RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.policy_fc1(x)
        actor_x = self.drop1(actor_x)
        actor_x = self.policy_fc2(actor_x)
        actor_x = self.drop2(actor_x)
        actor_x = self.policy_fc3(actor_x)
        actor_x = self.drop3(actor_x)
        actor_x = self.policy_fc4(actor_x)
        policy = self.actor_out(actor_x)

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return policy, value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, action_size):
        self.render = False

        # 행동의 크기 정의
        self.action_size = action_size

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.0007

        # 정책신경망과 가치신경망 생성
        self.model = A2C(self.action_size)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=5.0)
        self.states, self.actions, self.rewards = [], [], []
        self.state = None

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        self.state = self.flatter(state)
        policy = self.model(self.state)[0]
        policy = np.array(policy[0])

        mapInfo = state['mapInfo']
        mapInfo.extend(state['mapInfo'])
        reRollCnt = state['reRoll'][0]

        lst = [policy[i] if mapInfo[i][0] not in [TileType.BROKENTILE.value, TileType.DISTORTEDTILE.value, -1] else 0
               for i in range(len(policy)-1)]
        lst.append(policy[len(policy)-1] if reRollCnt > 0 else 0)

        if np.sum(lst) != 0:
            lst = [i / np.sum(lst) for i in lst]
        else:
            lst = [1 if mapInfo[i][0] not in [TileType.BROKENTILE.value, TileType.DISTORTEDTILE.value, -1] else 0
                   for i in range(len(lst)-1)]
            lst.append(1 if reRollCnt > 0 else 0)
            lst = [i / np.sum(lst) for i in lst]

        return np.random.choice(self.action_size, 1, p=lst)[0]

    def flatter(self, state):
        info = []
        for key in list(state.keys()):
            if type(state[key][0]) == list:
                info += sum(state[key], [])
            else:
                info += state[key]

        info = np.reshape(info, [1, len(info)])
        return info

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, action, reward, next_state, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            policy, value = self.model(self.state)
            _, next_value = self.model(self.flatter(next_state))
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 정책 신경망 오류 함수 구하기
            one_hot_action = tf.one_hot([action], self.action_size)
            action_prob = tf.reduce_sum(one_hot_action * policy, axis=1)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            advantage = tf.stop_gradient(target - value[0])
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            # 가치 신경망 오류 함수 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 오류 함수로 만들기
            # tf.Tensor(-0.15758774, shape=(), dtype=float32)
            loss = 0.2 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(loss)


keepLearning = False

if __name__ == "__main__":
    env = Env(render_speed = 0.001)
    env.mapIdx = 0

    state_size = env.state_size
    action_size = env.action_size
    mapSize = len(sum(env.action_space, []))

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(action_size)
    if keepLearning:
        agent.model.load_weights('Save_model/PPOModel')

    tmpLosses, tmpPlayTimes = [], []
    lossGraph, playtimeGraph = [], []
    episodes = []
    ran = 5

    EPISODES = 100000
    STAGEUPDATER = int(EPISODES / 10)
    SAVEMODEL = 100
    SAVEGRAPH = 10

    for e in range(EPISODES):
        done = False
        score = 0
        loss_list = []

        state = env.reset()

        while not done:
            action = agent.get_action(state)
            nextState, reward, done = env.Action(action)

            loss = agent.train_model(action, reward, nextState, done)
            loss_list.append(loss)

            score += reward
            state = nextState

            if done:
                print("episode: {:3d} | score: {:.3f} | loss : {:.3f} | playTime : {:3d} | RerollCnt : {:3d}".format(
                      e, score, np.mean(loss_list), env.playTime, env.reRoll))

                episodes.append(e)
                tmpLosses.append(np.mean(loss_list))
                tmpPlayTimes.append(env.playTime)

                env.mapIdx = np.random.choice(range(0, ran), 1)[0]

        if e % SAVEGRAPH == SAVEGRAPH-1:
            lossGraph.append(np.mean(tmpLosses))
            playtimeGraph.append(np.mean(tmpPlayTimes))
            tmpLosses, tmpPlayTimes = [], []

            fig, ax1 = plt.subplots()
            ax1.plot(playtimeGraph, 'r')
            ax1.set_ylabel('PlayTime', color='red')
            ax2 = ax1.twinx()
            ax2.plot(lossGraph, 'b')
            ax2.set_ylabel('Loss', color='blue')
            plt.savefig("./Save_graph/Graph.png")
            plt.close()

            plt.plot(playtimeGraph, 'r')
            plt.xlabel("episode")
            plt.ylabel("PlayTime")
            plt.savefig("./Save_graph/Graph_PlayTime.png")
            plt.close()

            plt.plot(lossGraph, 'b')
            plt.xlabel("episode")
            plt.ylabel("Loss")
            plt.savefig("./Save_graph/Graph_Loss.png")
            plt.close()

        if e % SAVEMODEL == SAVEMODEL-1:
            agent.model.save_weights('Save_model/PPOModel', save_format='tf')

        if e % STAGEUPDATER == STAGEUPDATER-1:
            if ran < len(list(MapType)):
                ran += 5
