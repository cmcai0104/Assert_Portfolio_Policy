import pandas as pd
from environment.DNN_MarketEnv import DNNMarketEnv
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('data/create_feature.csv', index_col=0, header=0)
df = df.set_index('trade_date')
df = df.dropna(axis=0, how='any')

env = DNNMarketEnv(df)

model = keras.models.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_shape =(df.shape[1], )),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=env.action_dim, activation='sigmoid'),
])

loss_func = keras.losses.MSE()
optimizer = keras.optimizers.Adam()

train_loss = keras.metrics.MSE(name='train_loss')
test_loss = tf.keras.metrics.MSE(name='test_loss')

@tf.function
def train_step(x, y, model):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_func(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def test_step(x, y, model):
  predictions = model(x)
  loss = loss_func(y, predictions)
  test_loss(loss)


EPOCHS = 100
for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))