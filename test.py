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
    return tf.convert_to_tensor(discounted_rewards)-10000.


def train(model, obs, current_price, next_price, shares_held, sell_fee=0., buy_fee=0.015):
    with tf.GradientTape() as tape:
        current_price = tf.concat([current_price, tf.ones((current_price.shape[0], 1))], axis=1)
        next_price = tf.concat([next_price, tf.ones((next_price.shape[0], 1))], axis=1)
        asserts_worth = current_price * shares_held
        net_worth = tf.reduce_sum(asserts_worth, axis=1, keepdims=True)
        hold_rate = (asserts_worth / net_worth)
        actions = model(obs)
        target_rate = (actions / tf.reduce_sum(actions, keepdims=True, axis=1))
        para = np.zeros((actions.shape[0], actions.shape[1]-1), dtype=np.float32)
        sell_index = np.where((hold_rate[:,:-1] > target_rate[:,:-1]).numpy())
        buy_index = np.where((hold_rate[:,:-1] < target_rate[:,:-1]).numpy())
        para[sell_index] = 1 - sell_fee
        para[buy_index] = 1 / (1 - buy_fee)
        para = tf.constant(para)
        net_reduce = (tf.reduce_sum(hold_rate[:,:-1]*para, axis=1) + hold_rate[:,-1]) / (tf.reduce_sum(target_rate[:,:-1]*para, axis=1) + target_rate[:,-1])
        new_net = tf.reshape(net_reduce, (net_reduce.shape[0], 1)) * net_worth
        shares_held = new_net * target_rate / current_price
        reward = tf.reduce_sum(next_price*shares_held, axis=1)
        loss = discount_reward(reward)
    gradients = tape.gradient(actions, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_loop():
    ob = env.reset()
    obs = []
    current_prices = []
    next_prices = []
    shares_helds = []
    rewards = []
    while True:
        action = model(ob)
        ob, reward, done, (current_price, next_price, shares_held) = env.step(action)
        obs.append(ob[0])
        current_prices.append(current_price)
        next_prices.append(next_price)
        shares_helds.append(shares_held)
        rewards.append(reward)
        if done:
            break
    env.render()
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    current_prices = tf.convert_to_tensor(current_prices, dtype=tf.float32)
    next_prices = tf.convert_to_tensor(next_prices, dtype=tf.float32)
    shares_helds = tf.convert_to_tensor(shares_helds, dtype=tf.float32)
    train(model, obs, current_prices, next_prices, shares_helds, sell_fee=0., buy_fee=0.015)
    #history = model.fit(x=obs, y=model.predict(obs), sample_weight=np.array(rewards))
    #return history

with tf.GradientTape() as tape:
    predict = model(obs)
gradient = tape.gradient(predict, model.trainable_variables)
gradient = [grad*np.sum(rewards) for grad in gradient]
optimizer.apply_gradients(zip(gradient, model.trainable_variables))


if __name__ == '__main__':
    Epochs = 1000
    for epoch in range(Epochs):
        train_loop()


