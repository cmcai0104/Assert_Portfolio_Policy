import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, activations, Input, models, optimizers
import tensorflow_probability as tfp
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


def create_model():
    input_layer = Input(shape=df.shape[1])
    hidden_dense_layer1 = layers.Dense(64, activation=activations.tanh)(input_layer)
    hidden_batch_layer1 = layers.BatchNormalization()(hidden_dense_layer1)
    hidden_dense_layer2 = layers.Dense(32, activation=activations.tanh)(hidden_batch_layer1)
    hidden_batch_layer2 = layers.BatchNormalization()(hidden_dense_layer2)
    mean_hidden_dense = layers.Dense(16, activation=activations.tanh)(hidden_batch_layer2)
    mean_hidden_batch = layers.BatchNormalization()(mean_hidden_dense)
    output_mean = layers.Dense(env.action_dim, activation=activations.sigmoid)(mean_hidden_batch)

    cov_hidden_dense = layers.Dense(16, activation=activations.tanh)(hidden_batch_layer2)
    cov_hidden_batch = layers.BatchNormalization()(cov_hidden_dense)
    output_cov = layers.Dense(env.action_dim, activation=activations.sigmoid)(cov_hidden_batch)


    model = models.Model(inputs=[input_layer], outputs=[output_mean, output_cov])
    return model


model = create_model()
optimizer = optimizers.RMSprop(0.001)


def discount_reward(rewards, r=0.04/250):
    ret_rewards = rewards.numpy()[-1] / rewards
    discount = np.array(list(reversed(range(1, len(ret_rewards) + 1))), dtype=float)
    discount_rewards = ret_rewards ** (250./discount)
    return tf.convert_to_tensor(discount_rewards-1.1, dtype=tf.float32)


def interaction_process():
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
    return obs, rewards


def train():
    obs, rewards = interaction_process()
    with tf.GradientTape() as tape:
        loss = -tf.reduce_sum(tf.transpose(tf.math.log(model(obs)[1]))*rewards)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss.numpy(), rewards


if __name__ == '__main__':
    Epochs = 10000
    losses = []
    for epoch in range(Epochs):
        loss, rewards = train()
        losses.append(loss)
        print('epochs:{}, loss:{}'.format(epoch, loss))
        if epoch % 1000 == 0:
            plot_df = df[price_col]
            portfolio = [10000]
            portfolio.extend(rewards)
            portfolio.append(np.nan)
            plot_df.loc[:, 'rewards'] = portfolio
            plot_df.to_csv('./data_for_analysis/%s_df.csv' % epoch)
    model.save('./model/policy_gradient.h5')

