import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, activations, Input, models, optimizers
from environment.DNN_MarketEnv import DNNMarketEnv


df = pd.read_csv('./data/create_feature.csv', index_col=0, header=0)
df['trade_date'] = df['trade_date'].astype('datetime64')
df = df[df['trade_date'] <= pd.datetime.strptime('20190809', '%Y%m%d')]
df = df.set_index('trade_date')
df = df.fillna(method='ffill', axis=0)
colnames = df.columns
colnames = colnames[[col[:6] != '399016' for col in colnames]]
df = df[colnames]
df = df.dropna(axis=0, how='any')
price_col = colnames[[col[-5:] == 'price' for col in colnames]]


env = DNNMarketEnv(df)
input_layer = Input(shape=df.shape[1])
hidden_dense_layer1 = layers.Dense(64, activation=activations.tanh)(input_layer)
hidden_batch_layer1 = layers.BatchNormalization()(hidden_dense_layer1)
hidden_dense_layer2 = layers.Dense(32, activation=activations.tanh)(hidden_batch_layer1)
hidden_batch_layer2 = layers.BatchNormalization()(hidden_dense_layer2)
hidden_dense_layer3 = layers.Dense(16, activation=activations.tanh)(hidden_batch_layer2)
hidden_batch_layer3 = layers.BatchNormalization()(hidden_dense_layer3)
output_para = layers.Dense(env.action_dim, activation=activations.sigmoid)(hidden_batch_layer3)
output_hidden = layers.Dense(4, activation=activations.tanh)(output_para)
output_eval = layers.Dense(1, activation=activations.sigmoid)(output_hidden)
model = models.Model(inputs=[input_layer], outputs=[output_para, output_eval])
optimizer = optimizers.RMSprop(0.001)


def discount_reward(rewards, r=0.04/250):
    ret_rewards = rewards.numpy()[-1] / rewards
    discount = np.array(list(reversed(range(1, len(ret_rewards) + 1))), dtype=float)
    discount_rewards = ret_rewards ** (250./discount)
    return tf.convert_to_tensor(discount_rewards-1.1, dtype=tf.float32)


def train(model, obs, rewards):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_sum(tf.transpose(tf.math.log(model(obs)[1]))*rewards)
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
        #if math.isnan(reward):
        #    break
        obs.append(ob[0])
        rewards.append(reward)
        if done:
            break
    env.render()
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    rewards = discount_reward(rewards)
    loss = train(model, obs, rewards)
    return loss.numpy(), rewards


if __name__ == '__main__':
    Epochs = 10000
    losses = []
    for epoch in range(Epochs):
        loss, rewards = train_loop()
        losses.append(loss)
        print('epochs:{}, loss:{}'.format(epoch, loss))
        if epoch % 1000 == 0:
            plot_df = df[price_col]
            portfolio = [10000]
            portfolio.extend(rewards)
            portfolio.append(np.nan)
            plot_df.loc[:, 'rewards'] = portfolio
            plot_df.to_csv('./data_for_analysis/%s_df.csv' % epoch)
        #if math.isnan(rewards[-1]) or math.isnan(loss):
        #    break
    model.save('./model/policy_gradient.h5')

