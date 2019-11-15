import math
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
colnames = df.columns
price_col = colnames[[col[-5:]=='close' for col in colnames]]


env = DNNMarketEnv(df)
input_layer = tf.keras.layers.Input(shape=df.shape[1])
hidden_dense_layer1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
hidden_batch_layer1 = tf.keras.layers.BatchNormalization()(hidden_dense_layer1)
# hidden_drop_layer1 = tf.keras.layers.Dropout(0.2)(hidden_batch_layer1)
hidden_dense_layer2 = tf.keras.layers.Dense(32, activation='relu')(hidden_batch_layer1)
hidden_batch_layer2 = tf.keras.layers.BatchNormalization()(hidden_dense_layer2)
# hidden_drop_layer2 = tf.keras.layers.Dropout(0.2)(hidden_batch_layer2)
hidden_dense_layer3 = tf.keras.layers.Dense(16, activation='relu')(hidden_batch_layer2)
hidden_batch_layer3 = tf.keras.layers.BatchNormalization()(hidden_dense_layer3)
# hidden_drop_layer3 = tf.keras.layers.Dropout(0.2)(hidden_dense_layer3)
output_para = tf.keras.layers.Dense(env.action_dim, activation='sigmoid')(hidden_batch_layer3)
output_eval = tf.keras.layers.Dense(1, activation='sigmoid')(output_para)
model = tf.keras.models.Model(inputs = [input_layer], outputs=[output_para, output_eval])
optimizer = tf.keras.optimizers.RMSprop(0.001)
#model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))


def discount_reward(rewards, r=0.04/250):
    rewards = rewards.numpy()
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0.
    for t in reversed(range(0, len(rewards))):
        running_add = running_add / (1+r) + rewards[t]
        discounted_rewards[t] = running_add
    discounted_rewards = discounted_rewards / np.array(list(reversed(range(1, len(discounted_rewards) + 1))))
    return 10000. - tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)


def train(model, obs, reward):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.math.log(model(obs)[1]))*(reward - 12500)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss


def train_loop():

    ob = env.reset()
    obs = []
    rewards = []
    while True:
        action = model(ob)[0]
        ob, reward, done, (current_price, next_price, shares_held) = env.step(action)
        if math.isnan(reward):
            break
        obs.append(ob[0])
        rewards.append(reward)
        if done:
            break
    env.render()
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    #rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    #rewards = discount_reward(rewards)
    loss = train(model, obs, rewards[-1])
    return loss.numpy(), rewards


if __name__ == '__main__':
    Epochs = 100000
    losses = []
    for epoch in range(Epochs):
        loss, rewards = train_loop()
        losses.append(loss)
        if epoch % 1000 == 0:
            print('epochs:{}, loss:{}'.format(epoch, loss))
            plot_df = df[price_col]
            portfolio = [10000]
            portfolio.extend(rewards)
            portfolio.append(np.nan)
            plot_df['rewards'] = portfolio
            plot_df.to_csv('./data_for_analysis/%s_df.csv'%epoch)
        if math.isnan(rewards[-1]) or math.isnan(loss):
            break
    model.save()

