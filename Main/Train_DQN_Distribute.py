from Environment import Env
from Options import *

global episode, score_avg, EPISODES, STAGEUPDATER
episode, score_avg = 0, 0

EPISODES = 210000
STAGEUPDATER = int(EPISODES / 10)

# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(256, activation='leaky_relu')
        self.drop1 = Dropout(0.2)
        self.fc2 = Dense(512, activation='leaky_relu')
        self.drop2 = Dropout(0.2)
        self.fc3 = Dense(512, activation='leaky_relu')
        self.drop3 = Dropout(0.2)
        self.fc4 = Dense(512, activation='leaky_relu')
        self.drop4 = Dropout(0.2)
        self.fc5 = Dense(256, activation='leaky_relu')
        self.drop5 = Dropout(0.2)
        self.fc6 = Dense(128, activation='leaky_relu')
        self.fc_out = Dense(action_size,
                            kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x);    x = self.drop1(x)
        x = self.fc2(x);    x = self.drop2(x)
        x = self.fc3(x);    x = self.drop3(x)
        x = self.fc4(x);    x = self.drop4(x)
        x = self.fc5(x);    x = self.drop5(x)
        x = self.fc6(x);
        q = self.fc_out(x)
        return q

class DQNAgent:
    def __init__(self, state_size, action_size):
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.00003
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.threads = 32

        # 모델과 타깃 모델 생성
        self.global_model = DQN(self.action_size)
        self.global_model.build(input_shape=(None, self.state_size))

        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.writer = tf.summary.create_file_writer('Summary/DQN_Distribute')

    def train(self):
        runners = [Runner(id=i, action_size=self.action_size,
                          global_model=self.global_model, optimizer=self.optimizer, writer=self.writer,
                          discount_factor=self.discount_factor, epsilon_min=self.epsilon_min,
                          epsilon=self.epsilon, epsilon_decay=self.epsilon_decay)
                   for i in range(self.threads)]

        for i, runner in enumerate(runners):
            print(f"Runner_{i} start")
            runner.start()

        while True:
            self.global_model.save_weights('Save_model/DQNModel_Distribute Version', save_format='tf')
            time.sleep(60 * 10)


# 카트폴 예제에서의 DQN 에이전트
class Runner(threading.Thread):
    def __init__(self, id, action_size, global_model, optimizer, writer, discount_factor,
                 epsilon_min, epsilon, epsilon_decay):
        threading.Thread.__init__(self)

        self.id = id
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 64
        self.train_start = 500

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.global_model = global_model
        self.optimizer = optimizer

        self.local_model = DQN(self.action_size)
        self.target_model = DQN(self.action_size)
        self.env = Env(render_speed = 0.001)
        self.writer = writer

        # 타깃 모델 초기화
        self.update_target_model()

        self.states, self.actions, self.reward = [], [], []
        self.stageLevel = 1

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.local_model.get_weights())

    def draw_tensorboard(self, score, loss, e):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=e)
            tf.summary.scalar('Mean loss/Episode', loss, step=e)

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        mapInfo = state['mapInfo'][:]
        mapInfo.extend(state['mapInfo'][:])
        reRollCnt = state['reRoll'][0]

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        else:
            state = self.flatter(state)
            q_value = self.local_model(state)[0]

            return np.argmax(q_value)


    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        state = self.flatter(state)
        next_state = self.flatter(next_state)
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        global_params = self.global_model.trainable_weights
        local_params = self.local_model.trainable_weights
        with tf.GradientTape() as tape:
            loss = self.getLoss()

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, local_params)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)
        self.optimizer.apply_gradients(zip(grads, global_params))
        self.local_model.set_weights(self.global_model.get_weights())

        return np.array(loss)

    def getLoss(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        # 현재 상태에 대한 모델의 큐함수
        predicts = self.local_model(states)
        one_hot_action = tf.one_hot(actions, self.action_size)
        predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

        # 다음 상태에 대한 타깃 모델의 큐함수
        target_predicts = self.target_model(next_states)
        target_predicts = tf.stop_gradient(target_predicts)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        max_q = np.amax(target_predicts, axis=-1)
        targets = rewards + (1 - dones) * self.discount_factor * max_q
        loss = tf.reduce_mean(tf.square(targets - predicts))

        return loss

    def flatter(self, state):
        info = []
        for key in list(state.keys()):
            if type(state[key][0]) == list:
                info += sum(state[key], [])
            else:
                info += state[key]

        info = np.reshape(info, [1, len(info)])
        return info

    def run(self):
        global episode, score_avg
        self.env.mapIdx = np.random.choice(range(0, self.stageLevel * 5), 1)[0]

        while episode < EPISODES:
            done = False
            score = 0
            lossList = []

            state = self.env.reset()

            while not done:
                action = self.get_action(state)
                nextState, reward, done = self.env.Action(action)

                self.append_sample(state, action, reward, nextState, done)

                if len(self.memory) >= self.train_start:
                    loss = self.train_model()
                    lossList.append(loss)

                score += reward
                state = nextState

                if done:
                    episode += 1
                    self.draw_tensorboard(score, np.mean(lossList), episode)
                    print("Actor : {:3d} | episode: {:3d} | score: {:.3f} | loss : {:.3f} | maxPlayTime : {:3d} | playTime : {:3d} | RerollCnt : {:3d}".format(
                            self.id, episode, score, np.mean(lossList), self.env.map.maxPlayTime, self.env.playTime, self.env.reRoll))

        if episode % STAGEUPDATER == STAGEUPDATER - 1:
            if self.stageLevel * 5 < len(list(MapType)):
                self.stageLevel += 1


if __name__ == "__main__":
    tmpEnv = Env()
    state_size = tmpEnv.state_size
    action_size = tmpEnv.action_size
    global_agent = DQNAgent(state_size=state_size, action_size=action_size)
    global_agent.train()