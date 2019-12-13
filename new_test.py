from environment.DNN_Distribution import DNNMarketEnv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, Input, layers, activations, optimizers
import tensorflow_probability as tfp


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


def create_normal_model():
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
    output_cov = layers.Dense(((env.action_dim + 1) * env.action_dim / 2), activation=activations.tanh)(cov_hidden_batch)

    model = models.Model(inputs=[input_layer], outputs=[output_mean, output_cov])
    return model


model = create_normal_model()
optimizer = optimizers.RMSprop(0.001)


def env_interaction():
    ob = env.reset()
    rewards = []
    dists = []
    actions = []
    while True:
        action = model(ob)
        ob, reward, done, (dist, target_rate) = env.step(action)
        rewards.append(reward)
        dists.append(dist)
        actions.append(target_rate)
        if done:
            break
    env.render()
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
    return rewards, log_probs

def discount_reward(rewards, r=0.04/250):
    ret_rewards = rewards.numpy()[-1] / rewards
    discount = np.array(list(reversed(range(1, len(ret_rewards) + 1))), dtype=float)
    discount_rewards = ret_rewards ** (250./discount)
    return tf.convert_to_tensor(discount_rewards-1.1, dtype=tf.float32)


def train():
    rewards, log_probs = env_interaction()
    discount_rewards = discount_reward(rewards)
    with tf.GradientTape() as tape:
        loss = -tf.reduce_sum(discount_rewards * tf.transpose(log_probs)[0][0])
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss.numpy(), rewards


if __name__ == '__main__':
    Epochs = 1000
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