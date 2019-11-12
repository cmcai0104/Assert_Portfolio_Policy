import numpy as np
import pandas as pd
from environment.DNN_MarketEnv import DNNMarketEnv
import tensorflow as tf


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
optimizer = tf.keras.optimizers.RMSprop()
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

def discount_reward(rewards, r=0.04/250):
    rewards = rewards.numpy()
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0.
    for t in reversed(range(0, len(rewards))):
        running_add = running_add / (1+r) + rewards[t]
        discounted_rewards[t] = running_add
    discounted_rewards = discounted_rewards / np.array(list(reversed(range(1, len(discounted_rewards) + 1))))
    return 10000. - tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)


def train(model, obs, rewards):
    rewards = tf.reshape(rewards, (rewards.shape[0], 1))
    with tf.GradientTape() as tape:
        predict = tf.math.log(model(obs)) * rewards
    gradient = tape.gradient(predict, model.trainable_variables)

    optimizer.apply_gradients(zip(gradient, model.trainable_variables))


def train_loop():
    ob = env.reset()
    obs = []
    rewards = []
    while True:
        action = model(ob)
        ob, reward, done, (current_price, next_price, shares_held) = env.step(action)
        obs.append(ob[0])
        rewards.append(reward)
        if done:
            break
    env.render()
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    rewards = discount_reward(rewards)
    train(model, obs, rewards)


if __name__ == '__main__':
    Epochs = 10000
    for epoch in range(Epochs):
        train_loop()


