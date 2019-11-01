

#初始化 Q-Table(Q表)
import pickle # 保存模型用
from collections import defaultdict
import gym  # 0.12.5
import numpy as np

# 默认将Action 0,1,2的价值初始化为0
Q = defaultdict(lambda: [0, 0, 0])

#连续状态映射
env = gym.make('MountainCar-v0')

def transform_state(state):
    """将 position, velocity 通过线性转换映射到 [0, 40] 范围内"""
    pos, v = state
    pos_low, v_low = env.observation_space.low
    pos_high, v_high = env.observation_space.high

    a = 40 * (pos - pos_low) / (pos_high - pos_low)
    b = 40 * (v - v_low) / (v_high - v_low)

    return int(a), int(b)

#更新 Q-Table
#开始训练

lr, factor = 0.7, 0.95
episodes = 10000  # 训练10000次
score_list = []  # 记录所有分数
for i in range(episodes):
    s = transform_state(env.reset())
    score = 0
    while True:
        a = np.argmax(Q[s])
        # 训练刚开始，多一点随机性，以便有更多的状态
        if np.random.random() > i * 3 / episodes:
            a = np.random.choice([0, 1, 2])
        # 执行动作
        next_s, reward, done, _ = env.step(a)
        next_s = transform_state(next_s)
        # 根据上面的公式更新Q-Table
        Q[s][a] = (1 - lr) * Q[s][a] + lr * (reward + factor * max(Q[next_s]))
        score += reward
        s = next_s
        if done:
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
env.close()

# 保存模型
with open('MountainCar-v0-q-learning.pickle', 'wb') as f:
    pickle.dump(dict(Q), f)
    print('model saved')

#测试模型
import time
import pickle
import gym
import numpy as np

# 加载模型
with open('MountainCar-v0-q-learning.pickle', 'rb') as f:
    Q = pickle.load(f)
    print('model loaded')

env = gym.make('MountainCar-v0')
s = env.reset()
score = 0
while True:
    env.render()
    time.sleep(0.01)
    # transform_state函数 与 训练时的一致
    s = transform_state(s)
    a = np.argmax(Q[s]) if s in Q else 0
    s, reward, done, _ = env.step(a)
    score += reward
    if done:
        print('score:', score)
        break
env.close()