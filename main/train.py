import os

import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import copy
import matplotlib.pyplot as plt
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class REINFORCE(tf.keras.Model):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.fc3 = Dense(128, activation='relu')
        self.fc_out = Dense(action_size, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        policy = self.fc_out(x)
        return policy


# 그리드월드 예제에서의 REINFORCE 에이전트
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # REINFORCE 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = REINFORCE(self.action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

    # 정책신경망으로 행동 선택
    def get_action(self, mapSize, state):
        policy = self.model(state)[0]
        policy = np.array(policy)

        lst = []
        mapInfo = state[0][:len(state[0])-1]
        reRollCnt = state[0][len(state[0])-1]
        for i in range(len(policy)):
            if i == len(policy)-1:
                if reRollCnt > 0:
                    lst.append(policy[i])
                else:
                    lst.append(0)
            elif mapInfo[i] != 0:
                lst.append(policy[i])
            else:
                lst.append(0)
        # 맵 선택 + 리롤
        # mapInfo = state[0][:mapSize]
        # reRollCnt = state[0][mapSize]
        # posSelectPolicy = policy[:mapSize+1]
        # lst = []
        # for i in range(len(posSelectPolicy)):
        #     if i == len(posSelectPolicy)-1:
        #         if reRollCnt > 0:
        #             lst.append(posSelectPolicy[i])
        #         else:
        #             lst.append(0)
        #     elif mapInfo[i] != 1 and mapInfo[i] != 2:
        #         lst.append(posSelectPolicy[i])
        #     else:
        #         lst.append(0)
        #
        # posSelectPolicy = lst/np.sum(lst)
        #
        # # 어떤 손을 고를지
        # handSelectPolicy = policy[-2:]
        # handSelectPolicy = handSelectPolicy / np.sum(handSelectPolicy)
        #
        # return np.random.choice(len(handSelectPolicy), 1, p=handSelectPolicy)[0], np.random.choice(len(posSelectPolicy), 1, p=posSelectPolicy)[0]
        return np.random.choice(self.action_size, 1, p=lst)[0]

    # 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        # 크로스 엔트로피 오류함수 계산
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            policies = self.model(np.array(self.states))
            actions = np.array(self.actions)
            action_prob = tf.reduce_sum(actions * policies, axis=1)
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            entropy = - policies * tf.math.log(policies)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
        return np.mean(entropy)


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed = 0.01)
    state_size = env.state_size
    action_size = env.action_size
    mapSize = len(sum(env.action_space, []))

    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes, playTimes = [], [], []

    EPISODES = 10000
    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            # 현재 상태에 대한 행동 선택
            # handAction, mapAction = agent.get_action(mapSize, state)
            action = agent.get_action(mapSize, state)

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
            nextState = np.reshape(nextState, [1, state_size])

            agent.append_sample(state, action, reward)
            score += reward

            state = nextState

            if done:
                # 에피소드마다 정책신경망 업데이트
                entropy = agent.train_model()
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:.3f} | entropy: {:.3f} | playTime : {:3d} | RerollCnt : {:3d}".format(
                      e, score, entropy, env.playTime, env.reRoll))

                scores.append(entropy)
                episodes.append(e)
                playTimes.append(env.playTime)

                fig, ax1 = plt.subplots()
                ax1.plot(episodes, playTimes, 'r')
                ax1.set_ylabel('playTime', color='red')
                ax2 = ax1.twinx()
                ax2.plot(episodes, scores, 'b')
                ax2.set_ylabel('score', color='blue')
                plt.savefig("./save_graph/graph.png")
                plt.close()

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')
