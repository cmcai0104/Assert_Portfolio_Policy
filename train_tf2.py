import os
import matplotlib.pyplot as plt
import tensorflow as tf
# 下载训练数据集
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
batch_size = 32
train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_fp, batch_size, column_names=column_names, label_name=label_name, num_epochs=1)
###########这些 Dataset 对象是可迭代的
#features, labels = next(iter(train_dataset))

#####将特征打包到一个数组中#########
def pack_features_vector(features, labels):
  features = tf.stack(list(features.values()), axis=1)
  return features, labels
train_dataset = train_dataset.map(pack_features_vector)
#features, labels = next(iter(train_dataset))


#构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])
# 训练模型
## 定义损失和梯度函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y):
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)

# 计算梯度优化模型
def grad(model, input, target):
    with tf.GradientTape() as tape:
        loss_value = loss(model, input, target)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

#创建优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
train_loss_results = []
train_accuracy_results = []

num_epochs = 201
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg(loss_value)
        epoch_accuracy(y, model(x))

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))
#可视化损失函数随时间推移而变化的情况
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

#评估模型的效果
##建立测试数据集
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url), origin=test_url)
test_dataset = tf.data.experimental.make_csv_dataset(test_fp, batch_size,
                                                     column_names=column_names,
                                                     label_name='species', num_epochs=1, shuffle=False)
test_dataset = test_dataset.map(pack_features_vector)
#根据测试数据集评估模型
test_accuracy = tf.keras.metrics.Accuracy()
for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)
#使用经过训练的模型进行预测
for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))





x = tf.random.normal((5,4))
y = tf.random.normal((5,1))
#构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(4,)),
])

'''
<tf.Variable 'dense_18/kernel:0' shape=(4, 1) dtype=float32, numpy=
array([[-0.61312824],
       [-0.94801086],
       [-0.9145787 ],
       [ 1.093276  ]], dtype=float32)>, 
<tf.Variable 'dense_18/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

loss_object = tf.keras.losses.mse
def loss(model, x, y):
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)
def grad(model, input, target):
    with tf.GradientTape() as tape:
        loss_value = loss(model, input, target)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
'''
(<tf.Tensor: id=169810, shape=(5,), dtype=float32, numpy=
array([ 1.565703  ,  2.0619576 , 11.674311  ,  2.1939662 ,  0.03749825], dtype=float32)>, 
[<tf.Tensor: id=169841, shape=(4, 1), dtype=float32, numpy=
array([[ 6.8775167],
       [-6.180906 ],
       [-1.3133967],
       [21.383528 ]], dtype=float32)>, <tf.Tensor: id=169840, shape=(1,), dtype=float32, numpy=array([3.853183], dtype=float32)>])
'''
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_value, grads = grad(model, x, y)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
'''
[<tf.Variable 'dense_18/kernel:0' shape=(4, 1) dtype=float32, numpy=
array([[-0.62312824],
       [-0.9380109 ],
       [-0.9045787 ],
       [ 1.083276  ]], dtype=float32)>, 
<tf.Variable 'dense_18/bias:0' shape=(1,) dtype=float32, numpy=array([-0.00999999], dtype=float32)>]

'''
train_loss_results = []
train_accuracy_results = []
num_epochs = 201
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for i in range(5):
        x = tf.reshape(x[i], shape=(1,-1))
        y = tf.reshape(y[i], shape=(1,-1))
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss_avg(loss_value)
        epoch_accuracy(y, model(x))
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))