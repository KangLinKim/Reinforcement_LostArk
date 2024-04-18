from Environment import Env
from Options import *


# 정책 신경망과 가치 신경망 생성
class A2C(tf.keras.Model):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.policy_fc1 = Dense(256, activation='leaky_relu')
        self.drop1 = Dropout(0.2)
        self.policy_fc2 = Dense(512, activation='leaky_relu')
        self.drop2 = Dropout(0.2)
        self.policy_fc3 = Dense(512, activation='leaky_relu')
        self.drop3 = Dropout(0.2)
        self.policy_fc4 = Dense(256, activation='leaky_relu')
        self.drop4 = Dropout(0.2)
        self.policy_fc5 = Dense(128, activation='leaky_relu')
        self.actor_out = Dense(action_size, activation='softmax',
                               kernel_initializer=RandomUniform(-1e-3, 1e-3))

        self.critic_fc1 = Dense(256, activation='leaky_relu')
        self.critic_fc2 = Dense(128, activation='leaky_relu')
        self.critic_fc3 = Dense(64, activation='leaky_relu')
        self.critic_out = Dense(1, kernel_initializer =RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.policy_fc1(x)
        actor_x = self.drop1(actor_x)
        actor_x = self.policy_fc2(actor_x)
        actor_x = self.drop2(actor_x)
        actor_x = self.policy_fc3(actor_x)
        actor_x = self.drop3(actor_x)
        actor_x = self.policy_fc4(actor_x)
        actor_x = self.drop4(actor_x)
        actor_x = self.policy_fc5(actor_x)
        policy = self.actor_out(actor_x)

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        critic_x = self.critic_fc3(critic_x)
        value = self.critic_out(critic_x)
        return policy, value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class PPOAgent:
    def __init__(self, action_size):
        self.render = False

        # 행동의 크기 정의
        self.action_size = action_size

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self._lambda = 0.95
        self.epsilon = 0.2
        self.batch_size = 128
        self.n_epoch = 3

        # 정책신경망과 가치신경망 생성
        self.model = A2C(self.action_size)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=5.0)
        self.states, self.actions, self.rewards = [], [], []
        self.state = None
        self.memory = list()

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        self.state = self.flatter(state)
        policy = self.model(self.state)[0]
        policy = np.array(policy[0])

        mapInfo = state['mapInfo'][:]
        mapInfo.extend(state['mapInfo'][:])
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
    def train_model(self):
        state      = [m[0] for m in self.memory]
        action     = [m[1] for m in self.memory]
        reward     = [m[2] for m in self.memory]
        next_state = [m[3] for m in self.memory]
        done       = [m[4] for m in self.memory]
        self.memory.clear()

        prob_old, value, next_value = [], [], []
        for st, nt in zip(state, next_state):
            _prob_old, _value = self.model(st)
            _, _next_value = self.model(nt)
            prob_old.append(_prob_old)
            value.append(_value)
            next_value.append(_next_value)

        delta = [_reward + (1 - _done) * self.discount_factor * _next_value - _value
                 for _reward, _done, _next_value, _value in zip(reward, done, next_value, value)]
        adv = delta[:]
        for t in reversed(range(n_step - 1)):
            adv[:t] += [(1-_done) * self.discount_factor * self._lambda * _adv
                          for _done, _adv in zip(done[:t], adv[:t+1])]

        ret = adv + value

        actor_losses, critic_losses = [], []
        idxs = np.arange(len(reward))
        for _ in range(self.n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                model_params = self.model.trainable_variables
                idx = idxs[offset: offset + self.batch_size]
                _state, _action, _ret, _adv, _prob_old = [], [], [], [], []
                for i in idx:
                    _state.append(state[i])
                    _action.append(action[i])
                    _ret.append(ret[i])
                    _adv.append(adv[i])
                    _prob_old.append(prob_old[i])

                grads, actor_loss, critic_loss = self.compute_loss(_state, _action, _ret, _adv, _prob_old)
                actor_losses.extend(actor_loss)
                critic_losses.extend(critic_loss)
                self.optimizer.apply_gradients(zip(grads, model_params))

        return np.mean(actor_losses), np.mean(critic_losses)

    def compute_loss(self, states, actions, rets, advantages, prob_olds):
        model_params = self.model.trainable_variables
        total_loss = 0
        actor_losses, critic_losses = [], []
        with tf.GradientTape() as tape:
            for _state, _action, _ret, _adv, _prob_old in zip(states, actions, rets, advantages, prob_olds):
                action_probs, value = self.model(_state)

                probs = tf.reduce_sum(tf.one_hot(_action, 1) * action_probs, axis=1)
                _prob_old = tf.convert_to_tensor(_prob_old)

                ratio = probs / (_prob_old + 1e-8)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
                surrogate1 = ratio * _adv
                surrogate2 = clipped_ratio * _adv
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                critic_loss = tf.reduce_mean(tf.square(_ret - value))

                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                total_loss += actor_loss + critic_loss
            total_loss /= len(states)
        grads = tape.gradient(tf.convert_to_tensor(total_loss), model_params)

        return grads, actor_losses, critic_losses

    def append_sample(self, state, action, reward, nextState, done):
        state = self.flatter(state)
        nextState = self.flatter(nextState)
        self.memory.append([state, action, reward, nextState, done])


if __name__ == "__main__":
    env = Env(render_speed = 0.001)
    env.mapIdx = 0

    state_size = env.state_size
    action_size = env.action_size
    mapSize = len(sum(env.action_space, []))

    keepLearning = False

    # 액터-크리틱(A2C) 에이전트 생성
    agent = PPOAgent(action_size)
    if keepLearning:
        agent.model.load_weights('Save_model/PPOModel')

    actorLosses, criticLosses, playTimes = [], [], []
    actorLossGraph, criticLossGraph, playtimeGraph = [], [], []
    episodes = []
    n_step = 5

    score_avg = 0
    ran = 5

    EPISODES = 100000
    STAGEUPDATER = int(EPISODES / 10)
    SAVEMODEL = 100
    SAVEGRAPH = 10

    for e in range(EPISODES):
        done = False
        score, step = 0, 0
        actorList, criticList = [], []

        state = env.reset()

        while not done:
            step += 1
            action = agent.get_action(state)
            nextState, reward, done = env.Action(action)

            agent.append_sample(state, action, reward, nextState, done)

            score += reward
            state = nextState

            if step % n_step == 0 or done:
                actor_loss, critic_loss = agent.train_model()
                actorList.append(actor_loss)
                criticList.append(critic_loss)

            if done:
                print("episode: {:3d} | score: {:.3f} | actorLoss : {:.3f} | criticLoss : {:.3f} | maxPlayTime : {:3d} | playTime : {:3d} | RerollCnt : {:3d}".format(
                      e, score, np.mean(actorList), np.mean(criticList), env.map.maxPlayTime, env.playTime, env.reRoll))

                episodes.append(e)

                actorLosses.append(np.mean(actorList))
                criticLosses.append(np.mean(criticList))
                playTimes.append(env.playTime)

                env.mapIdx = np.random.choice(range(0, ran), 1)[0]

        if e % SAVEGRAPH == SAVEGRAPH-1:
            actorLossGraph.append(np.mean(actorLosses))
            criticLossGraph.append(np.mean(criticLosses))
            playtimeGraph.append(np.mean(playTimes))
            actorLosses, criticLosses, playTimes = [], [], []

            plt.plot(playtimeGraph, 'r')
            plt.xlabel("episode")
            plt.ylabel("PlayTime")
            plt.savefig("./Save_graph/PPO/Graph_PlayTime.png")
            plt.close()

            plt.plot(actorLossGraph, 'b')
            plt.xlabel("episode")
            plt.ylabel("ActorLoss")
            plt.savefig("./Save_graph/PPO/Graph_actorLoss.png")
            plt.close()

            plt.plot(criticLossGraph, 'g')
            plt.xlabel("episode")
            plt.ylabel("CriticLoss")
            plt.savefig("./Save_graph/PPO/Graph_criticLoss.png")
            plt.close()

        if e % SAVEMODEL == SAVEMODEL-1:
            agent.model.save_weights('Save_model/Model', save_format='tf')

        if e % STAGEUPDATER == STAGEUPDATER-1:
            if ran < len(list(MapType)):
                ran += 5
