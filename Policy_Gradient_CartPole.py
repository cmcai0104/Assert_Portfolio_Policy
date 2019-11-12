#from warnings import simplefilter
#simplefilter(action='ignore', category='FutureWarning')

import numpy as np
from tensorflow.keras import models,layers,optimizers
import gym
import matplotlib.pyplot as plt

#搭建神经网络
env = gym.make('CartPole-v0')
STATE_DIM, ACTION_DIM = 4, 2
model = models.Sequential([
    layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(ACTION_DIM, activation="softmax")
])
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.001))

def choose_action(s):
    """预测动作"""
    prob = model.predict(np.array([s]))[0]
    return np.random.choice(len(prob), p=prob)

#优化策略
##衰减的累加期望
def discount_rewards(rewards, gamma=0.95):
    """计算衰减reward的累加期望，并中心化和标准化处理"""
    prior = 0
    out = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        prior = prior * gamma + rewards[i]
        out[i] = prior
    return out

##完成一次游戏后，根据整个过程记录的 [(state，action，reward)] 训练模型给
def train(records):
    # 过程所有state构成list
    s_batch = np.array([record[0] for record in records])
    # 对action进行 one-hot 编码
    a_batch = np.array([[1 if record[1] == i else 0 for i in range(ACTION_DIM)]
                        for record in records])
    #假设predict的概率是 [0.3, 0.7]，选择的动作是 [0, 1]
    #则动作[0, 1]的概率等于 [0, 0.7] = [0.3, 0.7] * [0, 1]
    prob_batch = model.predict(s_batch) * a_batch
    r_batch = discount_rewards([record[2] for record in records])
    model.fit(s_batch, prob_batch, sample_weight=r_batch, verbose=0)

#训练过程与结果
episodes = 2000  # 至多2000次
score_list = []  # 记录所有分数
for i in range(episodes):
    s = env.reset()
    score = 0
    replay_records = []
    while True:
        a = choose_action(s)
        next_s, r, done, _ = env.step(a)
        replay_records.append((s, a, r))

        score += r
        s = next_s
        if done:
            train(replay_records)
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
    # 最后10次的平均分大于 195 时，停止并保存模型
    if np.mean(score_list[-10:]) > 195:
        model.save('./model/CartPole-v0-pg.h5')
        break
env.close()

plt.plot(score_list)
x = np.array(range(len(score_list)))
smooth_func = np.poly1d(np.polyfit(x, score_list, 3))
plt.plot(x, smooth_func(x), label='Mean', linestyle='--')
plt.show()


#测试
import time
import numpy as np
import gym
from tensorflow.keras import models

saved_model = models.load_model('./model/CartPole-v0-pg.h5')
env = gym.make("CartPole-v0")

for i in range(5):
    s = env.reset()
    score = 0
    while True:
        time.sleep(0.01)
        env.render()
        prob = saved_model.predict(np.array([s]))[0]
        a = np.random.choice(len(prob), p=prob)
        s, r, done, _ = env.step(a)
        score += r
        if done:
            print('using policy gradient, score: ', score)  # 打印分数
            break
env.close()
