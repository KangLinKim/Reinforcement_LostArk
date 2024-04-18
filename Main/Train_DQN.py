from Environment import Env
from Options import *


# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(256, activation='leaky_relu')
        self.drop1 = Dropout(0.2)
        self.fc2 = Dense(512, activation='leaky_relu')
        self.drop2 = Dropout(0.2)
        self.fc3 = Dense(1028, activation='leaky_relu')
        self.drop3 = Dropout(0.2)
        self.fc4 = Dense(1028, activation='leaky_relu')
        self.drop4 = Dropout(0.2)
        self.fc5 = Dense(1028, activation='leaky_relu')
        self.drop5 = Dropout(0.2)
        self.fc6 = Dense(512, activation='leaky_relu')
        self.drop6 = Dropout(0.2)
        self.fc7 = Dense(256, activation='leaky_relu')
        self.drop7 = Dropout(0.2)
        self.fc8 = Dense(128, activation='leaky_relu')
        self.fc_out = Dense(action_size,
                            kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x);    x = self.drop1(x)
        x = self.fc2(x);    x = self.drop2(x)
        x = self.fc3(x);    x = self.drop3(x)
        x = self.fc4(x);    x = self.drop4(x)
        x = self.fc5(x);    x = self.drop5(x)
        x = self.fc6(x);    x = self.drop6(x)
        x = self.fc7(x);    x = self.drop7(x)
        x = self.fc8(x)
        q = self.fc_out(x)
        return q


# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = DQN(self.action_size)
        self.target_model = DQN(self.action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # 타깃 모델 초기화
        self.update_target_model()

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        mapInfo = state['mapInfo'][:]
        mapInfo.extend(state['mapInfo'][:])
        reRollCnt = state['reRoll'][0]

        if np.random.rand() <= self.epsilon:
            lst = [1 if mapInfo[i][0] not in [TileType.BROKENTILE.value, TileType.DISTORTEDTILE.value, -1] else 0
                   for i in range(len(mapInfo))]
            lst.append(1 if reRollCnt > 0 else 0)
            lst = [i / np.sum(lst) for i in lst]
            return np.random.choice(self.action_size, 1, p=lst)[0]

        else:
            state = self.flatter(state)
            q_value = self.model(state)[0]

            lst = [q_value[i] if mapInfo[i][0] not in [TileType.BROKENTILE.value, TileType.DISTORTEDTILE.value, -1] else 0
                for i in range(len(q_value) - 1)]
            lst.append(q_value[len(q_value) - 1] if reRollCnt > 0 else 0)

            if np.sum(lst) != 0:
                lst = [i / np.sum(lst) for i in lst]
                return np.argmax(lst)
            else:
                lst = [1 if mapInfo[i][0] not in [TileType.BROKENTILE.value, TileType.DISTORTEDTILE.value, -1] else 0
                       for i in range(len(lst) - 1)]
                lst.append(1 if reRollCnt > 0 else 0)
                lst = [i / np.sum(lst) for i in lst]
                return np.random.choice(self.action_size, 1, p=lst)[0]


    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        state = self.flatter(state)
        next_state = self.flatter(next_state)
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        # 학습 파라메터
        model_params = self.model.trainable_weights
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 상태에 대한 타깃 모델의 큐함수
            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            # 벨만 최적 방정식을 이용한 업데이트 타깃
            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(loss)

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
    env = Env(render_speed = 0.001)
    env.mapIdx = 0

    state_size = env.state_size
    action_size = env.action_size
    mapSize = len(sum(env.action_space, []))

    keepLearning = False

    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)
    if keepLearning:
        agent.model.load_weights('Save_model/DQNModel')

    tmpLosses, tmpPlayTimes = [], []
    LossGraph, playtimeGraph = [], []
    episodes = []
    score_avg = 0
    ran = 5

    EPISODES = 100000
    # global episode,
    STAGEUPDATER = int(EPISODES / 10)
    SAVEMODEL = 100
    SAVEGRAPH = 10
    for e in range(EPISODES):
        done = False
        score = 0
        loss_list, score_list = [], []

        state = env.reset()

        while not done:
            action = agent.get_action(state)
            nextState, reward, done = env.Action(action)

            agent.append_sample(state, action, reward, nextState, done)

            if len(agent.memory) >= agent.train_start:
                loss = agent.train_model()
                loss_list.append(loss)

            score += reward
            state = nextState

            if done:
                print("episode: {:3d} | score: {:.3f} | loss : {:.3f} | maxPlayTime : {:3d} | playTime : {:3d} | RerollCnt : {:3d}".format(
                    e, score, np.mean(loss_list), env.map.maxPlayTime, env.playTime, env.reRoll))

                episodes.append(e)
                tmpLosses.append(np.mean(loss_list))
                tmpPlayTimes.append(env.playTime)

                env.mapIdx = np.random.choice(range(0, ran), 1)[0]

        if e % SAVEGRAPH == SAVEGRAPH - 1:
            LossGraph.append(np.mean(tmpLosses))
            playtimeGraph.append(np.mean(tmpPlayTimes))
            tmpLosses, tmpPlayTimes = [], []

            fig, ax1 = plt.subplots()
            ax1.plot(playtimeGraph, 'r')
            ax1.set_ylabel('PlayTime', color='red')
            ax2 = ax1.twinx()
            ax2.plot(LossGraph, 'b')
            ax2.set_ylabel('Loss', color='blue')
            plt.savefig("./Save_graph/DQN/Graph.png")
            plt.close()

            plt.plot(playtimeGraph, 'r')
            plt.xlabel("episode")
            plt.ylabel("PlayTime")
            plt.savefig("./Save_graph/DQN/Graph_PlayTime.png")
            plt.close()

            plt.plot(LossGraph, 'b')
            plt.xlabel("episode")
            plt.ylabel("Loss")
            plt.savefig("./Save_graph/DQN/Graph_Loss.png")
            plt.close()

        if e % SAVEMODEL == SAVEMODEL - 1:
            agent.model.save_weights('Save_model/DQNModel', save_format='tf')

        if e % STAGEUPDATER == STAGEUPDATER - 1:
            if ran < len(list(MapType)):
                ran += 5