import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, activations, Input, models, optimizers, losses
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


def create_mode():
    input_layer = Input(shape=df.shape[1])
    hidden_dense_layer1 = layers.Dense(64, activation=activations.tanh)(input_layer)
    hidden_batch_layer1 = layers.BatchNormalization()(hidden_dense_layer1)
    hidden_dense_layer2 = layers.Dense(32, activation=activations.tanh)(hidden_batch_layer1)
    hidden_batch_layer2 = layers.BatchNormalization()(hidden_dense_layer2)
    hidden_dense_layer3 = layers.Dense(16, activation=activations.tanh)(hidden_batch_layer2)
    hidden_batch_layer3 = layers.BatchNormalization()(hidden_dense_layer3)
    output_vector = layers.Dense(env.action_dim, activation=activations.sigmoid)(hidden_batch_layer3)
    output_matrix_dense_layer1 = layers.Dense(32, activation=activations.tanh)(hidden_batch_layer3)
    output_matrix_batch_layer1 = layers.BatchNormalization()(output_matrix_dense_layer1)
    output_matrix_dense_layer2 = layers.Dense(64, activation=activations.tanh)(output_matrix_batch_layer1)
    output_matrix_batch_layer2 = layers.BatchNormalization()(output_matrix_dense_layer2)
    output_matrix = layers.Reshape(target_shape=(env.action_dim, env.action_dim))(output_matrix_batch_layer2)
    output_scalar_dense_layer = layers.Dense(4, activation=activations.tanh)(hidden_batch_layer3)
    output_scalar = layers.Dense(1)(output_scalar_dense_layer)
    model = models.Model(inputs=[input_layer], outputs=[output_vector, output_matrix, output_scalar])
    return model


Q_model = create_mode()
optimizer = optimizers.RMSprop(0.001)
delay_model = Q_model


def discount_rewards(rewards, r=0.04):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add / (1+r/250) + rewards[t]
        discounted_rewards[t] = running_add
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    return discounted_rewards


def train(Q_model, delay_model, obs, rewards, action):
    with tf.GradientTape() as tape:
        predict = Q_model(obs)
        qvalue = (tf.expand_dims((action - predict[0]),1) @ \
                 (tf.transpose(predict[1], perm=[0,2,1]) @ predict[1]) @ \
                 tf.expand_dims((action - predict[0]),2) + tf.expand_dims(predict[2],2))
        qvalue = (tf.squeeze(qvalue) - rewards)[:-1]
        expect_qvalue = tf.squeeze(delay_model(obs)[2])[1:]
        loss = losses.MSE(y_pred=qvalue, y_true=expect_qvalue)
    gradient = tape.gradient(loss, Q_model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, Q_model.trainable_variables))
    return loss


def train_loop():
    ob = env.reset()
    obs = []
    rewards = []
    actions = []
    while True:
        action = Q_model(ob)[0]
        ob, reward, done, (current_price, next_price, shares_held) = env.step(action)
        obs.append(ob[0])
        rewards.append(reward)
        actions.append(action)
        if done:
            break
    env.render()
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    rewards = discount_rewards(rewards)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)[:,0,:]
    loss = train(Q_model, delay_model, obs, rewards, actions)
    return loss.numpy(), rewards


if __name__ == '__main__':
    Epochs = 10000
    losses = []
    for epoch in range(Epochs):
        loss, rewards = train_loop()
        losses.append(loss)
        print('epochs:{}, loss:{}'.format(epoch, loss))
        if epoch % 100 == 0:
            delay_model = Q_model
        if epoch % 1000 == 0:
            plot_df = df[price_col]
            portfolio = [10000]
            portfolio.extend(rewards)
            portfolio.append(np.nan)
            plot_df.loc[:, 'rewards'] = portfolio
            plot_df.to_csv('./data_for_analysis/q_learn/%s_df.csv' % epoch)
    Q_model.save('./model/q_learning.h5')

