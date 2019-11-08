import numpy as np
import pandas as pd
from environment.DNN_MarketEnv import DNNMarketEnv
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv('./data/create_feature.csv', index_col=0, header=0)
df['trade_date'] = df['trade_date'].astype('datetime64')
df = df[df['trade_date'] <= pd.datetime.strptime('20190809', '%Y%m%d')]
df = df.set_index('trade_date')
df = df.fillna(method='ffill', axis=0)
df = df.dropna(axis=0, how='any')

env = DNNMarketEnv(df)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape =(df.shape[1],)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=env.action_dim, activation='sigmoid'),
])
model.compile(loss='mse', optimizer='rmsprop')
optimizer = tf.keras.optimizers.RMSprop()


def discount_reward(rewards, r=0.04/250):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add / (1+r) + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


def train_loop():
    ob = env.reset()
    obs = []
    rewards = []
    actions = []
    while True:
        action = model(ob)
        ob, reward, done, _ = env.step(action)
        obs.append(ob)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
    env.render()
    disc_rewards = discount_reward(rewards)
    avg_disc_rewards = disc_rewards / np.array(list(reversed(range(1, len(disc_rewards) + 1))))
    weight = (avg_disc_rewards.mean() - avg_disc_rewards)/avg_disc_rewards.std()
    model.fit(x=np.array(obs).reshape(-1,112), y=np.array(actions).reshape(-1,8), sample_weight=np.array(weight))


def plot_rewards(rewards, df):
    plt.plot(df.index[-len(rewards):], rewards)
    plt.show()

if __name__ == '__main__':
    Epochs = 100
    for epoch in range(Epochs):
        train_loop()

