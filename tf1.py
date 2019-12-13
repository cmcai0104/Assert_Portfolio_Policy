import tensorflow.compat.v1 as tf1
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.3)
del x_train_all, y_train_all

hidden_units = [100, 100]
class_num = 10
x = tf1.placeholder(tf.float32, [None, 28*28])
y = tf1.placeholder(tf.int64, [None])

input_for_next_layer = x
for hidden_unit in hidden_units:
    input_for_next_layer = tf1.layers.dense(input_for_next_layer, hidden_unit, activation=tf1.nn.relu)
logits = tf1.layers.dense(input_for_next_layer, class_num)
loss = tf1.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
train_op = tf1.train.AdamOptimizer(1e-3).minimize(loss)


init = tf1.global_variables_initializer()
batch_size = 20
epochs = 20
train_steps_per_epoch = x_train.shape[0] // batch_size
valid_steps = x_valid.shape[0] // batch_size

def eval_with_sess(sess, x, y, accuracy, images, labels, batch_size):
    eval_steps = images.shape[0] // batch_size
    eval_accuracies = []
    for step in range(eval_steps):
        batch_data = images[step*batch_size : (step+1)*batch_size]
        batch_label = labels[step*batch_size : (step+1)*batch_size]
        accuracy_val = sess.run(accuracy, feed_dict={x:batch_data, y:batch_label})
        eval_accuracies.append(accuracy_val)
    return np.mean(eval_accuracies)


with tf1.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            batch_data = x_train[step*batch_size : (step+1)*batch_size]
            batch_label = y_train[step*batch_size : (step+1)*batch_size]
            loss_val, accuracy_val, _ = sess.run([loss, accuracy, train_op], feed_dict={x: batch_data, y:batch_label})
            print('\r[Train] epoch: %d, step: %d, loss: %3.5f, accuracy:%2.2f'%(epoch, step, loss_val, accuracy_val), end="")
        valid_accuracy = eval_with_sess(sess, x, y, accuracy, x_valid, y_valid, batch_size)
        print("\t[Valid] acc: %2.2f"%(valid_accuracy))


