import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import gym
import pylab
import numpy as np
from environment import Env
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from Options import *

# 정책 신경망과 가치 신경망 생성
class A2C(tf.keras.Model):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.policy_fc1 = Dense(128, activation='leaky_relu')
        self.policy_fc2 = Dense(256, activation='leaky_relu')
        self.policy_fc3 = Dense(128, activation='leaky_relu')
        self.actor_out = Dense(action_size, activation='softmax',
                               kernel_initializer=RandomUniform(-1e-3, 1e-3))

        self.critic_fc1 = Dense(128, activation='selu')
        self.critic_fc2 = Dense(64, activation='leaky_relu')
        self.critic_out = Dense(1, kernel_initializer =RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.policy_fc1(x)
        actor_x = self.policy_fc2(actor_x)
        actor_x = self.policy_fc3(actor_x)
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

        lst = [policy[i] if mapInfo[i][2] != TileType.BROKENTILE.value and mapInfo[i][2] != TileType.DISTORTEDTILE.value else 0
               for i in range(len(policy)-1)]
        lst.append(policy[len(policy)-1] if reRollCnt > 0 else 0)

        if np.sum(lst) != 0:
            lst = [i / np.sum(lst) for i in lst]
        else:
            lst = [1 if mapInfo[i][2] != TileType.BROKENTILE.value and mapInfo[i][2] != TileType.DISTORTEDTILE.value else 0
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
            loss = 0.2 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(loss)


if __name__ == "__main__":
    env = Env(render_speed = 0.001)

    state_size = env.state_size
    action_size = env.action_size
    mapSize = len(sum(env.action_space, []))

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(action_size)

    tmp1, tmp2 = 0, 0
    tmp3, tmp4 = [], []
    scores, episodes, playTimes = [], [], []
    score_avg = 0

    EPISODES = 100000
    for e in range(EPISODES):
        done = False
        score = 0
        loss_list = []
        state = env.reset()

        while not done:
            action = agent.get_action(state)

            if action >= mapSize*2:
                # 리롤
                handAction = 2
                mapAction = 0
            elif action >= mapSize:
                # 오른손 선택
                handAction = 1
                mapAction = action - mapSize
            else:
                # 왼손
                handAction = 0
                mapAction = action

            nextState, reward, done = env.Action(handAction, mapAction)

            loss = agent.train_model(action, reward, nextState, done)
            loss_list.append(loss)

            score += reward

            state = nextState

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:.3f} | loss : {:.3f} | playTime : {:3d} | RerollCnt : {:3d}".format(
                      e, score, np.mean(loss_list), env.playTime, env.reRoll))

                scores.append(np.mean(loss_list))
                episodes.append(e)
                playTimes.append(env.playTime)

                tmp1 += np.mean(loss_list)
                tmp2 += env.playTime

                # fig, ax1 = plt.subplots()
                # ax1.plot(episodes, playTimes, 'r')
                # ax1.set_ylabel('playTime', color='red')
                # ax2 = ax1.twinx()
                # ax2.plot(episodes, scores, 'b')
                # ax2.set_ylabel('score', color='blue')
                # plt.savefig("./save_graph/graph.png")
                # plt.close()

        # 100 에피소드마다 모델 저장
        if e % 10 == 9:
            tmp3.append(tmp2 / 10)
            tmp4.append(tmp1 / 10)
            tmp1, tmp2 = 0, 0

            fig, ax1 = plt.subplots()
            ax1.plot(tmp3, 'r')
            ax1.set_ylabel('PlayTime', color='red')
            ax2 = ax1.twinx()
            ax2.plot(tmp4, 'b')
            ax2.set_ylabel('Loss', color='blue')
            plt.savefig("./save_graph/graph.png")
            plt.close()

            plt.plot(tmp3, 'r')
            plt.xlabel("episode")
            plt.ylabel("PlayTime")
            plt.savefig("./save_graph/graph_PlayTime.png")
            plt.close()

            plt.plot(tmp4, 'b')
            plt.xlabel("episode")
            plt.ylabel("Loss")
            plt.savefig("./save_graph/graph_Loss.png")
            plt.close()

        if e % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')