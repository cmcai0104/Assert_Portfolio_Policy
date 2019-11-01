import numpy as np
from environments.MarketEnv import MarketEnv
from tensorflow.keras import models,layers,optimizers
import matplotlib.pyplot as plt
import gym

env = MarketEnv()
STATE_DIM, ACTION_DIM = 10, 8
models = models.Sequential([
    layers.Dense(10, activation='relu', input_dim=STATE_DIM),
    layers.Dropout(0.1),
    layers.Dense(ACTION_DIM, activation='softmax')
])

models.compile(loss='MSE', optimizer=optimizers.Adam(0.001))

def action(s):
    pass

while True:
    s = env.reset()
    score = 0
    replay_records = []
    while True:
        # 互动记录(state, action, reward)
        a = action(s)
        next_s, r, done, _ = env.step(a)
        replay_records.append((s, a, r))
        if done:
            #train model
            pass
            # 最后10次的平均分大于 195 时，停止并保存模型
    if np.mean(score_list[-10:]) > 195:
        model.save('./model/CartPole-v0-pg.h5')
        break
env.close()

