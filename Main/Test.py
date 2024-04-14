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


# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
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


# 그리드월드 예제에서의 REINFORCE 에이전트
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

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        self.state = self.flatter(state)
        policy = self.model(self.state)[0]
        policy = np.array(policy[0])

        mapInfo = state['mapInfo']
        mapInfo.extend(state['mapInfo'])
        reRollCnt = state['reRoll'][0]

        lst = [policy[i] if mapInfo[i][2] not in [TileType.BROKENTILE.value, TileType.DISTORTEDTILE.value, -1] else 0
               for i in range(len(policy)-1)]
        lst.append(policy[len(policy)-1] if reRollCnt > 0 else 0)

        if np.sum(lst) != 0:
            lst = [i / np.sum(lst) for i in lst]
        else:
            lst = [1 if mapInfo[i][2] not in [TileType.BROKENTILE.value, TileType.DISTORTEDTILE.value, -1] else 0
                   for i in range(len(lst)-1)]
            lst.append(1 if reRollCnt > 0 else 0)
            lst = [i / np.sum(lst) for i in lst]

        return np.random.choice(self.action_size, 1, p=lst)[0]
        # return lst.index(max(lst))
    def flatter(self, state):
        info = []
        for key in list(state.keys()):
            if type(state[key][0]) == list:
                info += sum(state[key], [])
            else:
                info += state[key]

        info = np.reshape(info, [1, len(info)])
        return info


if __name__ == "__main__":
    # 환경과 에이전트 생성
    scoreList = []
    mapIdx = 0

    env = Env(render_speed = 0.001, _mapIdx=mapIdx)

    state_size = env.state_size
    action_size = env.action_size
    mapSize = len(sum(env.action_space, []))

    agent = A2CAgent(action_size)
    agent.model.load_weights('Save_model/Model')
    # scoreList = [[] for i in range(0, len(MapType))]

    EPISODES = 100000
    for e in range(EPISODES):
        done = False
        score = 0
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

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            nextState, reward, done = env.Action(handAction, mapAction)
            state = nextState

            if done:
                scoreList.append(env.playTime)

        if e % 2500 == 2499:
            cnt = 0
            for i in scoreList:
                if i <= env.map.maxPlayTime:
                    cnt += 1

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(scoreList, label='Playtime', color='#80B3FF')
            ax.set_xlabel('episode')
            ax.set_ylabel('PlayTime')

            ax.axhline(env.map.maxPlayTime, label='3 star', color='g')
            ax.axhline(env.map.maxPlayTime + 1, label='2 star', color='b')
            ax.axhline(env.map.maxPlayTime + 3, label='1 star', color='r')

            ax.legend(loc='upper left')
            plt.savefig(f"./Save_graph/Test/{list(MapType)[mapIdx].name}.png")
            plt.close()

            print(f'Mean : {np.mean(scoreList)}')

            scoreList = []
            mapIdx += 1
            env.mapIdx = mapIdx

            if mapIdx == len(MapType):
                break