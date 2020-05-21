from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from AlexNet_LRN import LRNModel, Parameter
import os
import numpy as np

(train_images, raw_train_labels), (test_images, raw_test_labels) = tf.keras.datasets.mnist.load_data()
# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

# one-hot
train_labels = tf.one_hot(raw_train_labels.tolist(), 10, axis=-1).numpy()
test_labels = tf.one_hot(raw_test_labels.tolist(), 10, axis=-1).numpy()
# model initialization
tf.compat.v1.disable_eager_execution()  # disable execution before Session()
x, _x, y = LRNModel.model_init([None, 28, 28],
                               [-1, 28, 28, 1],
                               [None, 10])
# layer1 Convolution 输入 卷积 ReLU LRN maxpooling
c1filter = Parameter.kernel([3, 3, 1, 64], 0.1)
conv1 = LRNModel.convolution(_x, c1filter, [1, 1])  # (26, 26, 64)
relu1 = LRNModel.relu(conv1)
lrn1 = LRNModel.local_response_normalization(relu1, 4, 1.0, 1e-3, 0.75)
maxpool1 = LRNModel.max_pooling(lrn1, [4, 4], [2, 2])  # (12, 12, 64)
# layer2 Convolution 卷积 ReLU LRN maxpooling
c2filter = Parameter.kernel([3, 3, 64, 64], 0.1)
conv2 = LRNModel.convolution(maxpool1, c2filter, [1, 1])  # (10, 10, 64)
relu2 = LRNModel.relu(conv2)
lrn2 = LRNModel.local_response_normalization(relu2, 4, 1.0, 1e-3, 0.75)
maxpool2 = LRNModel.max_pooling(lrn2, [3, 3], [1, 1])  # (8, 8, 64)
# layer3 卷积 ReLU
c3filter = Parameter.kernel([2, 2, 64, 128], 0.1)
conv3 = LRNModel.convolution(maxpool2, c3filter, [1, 1])  # (7, 7, 128)
relu3 = LRNModel.relu(conv3)
# layer4 卷积 ReLU
c4filter = Parameter.kernel([2, 2, 128, 128], 0.1)
conv4 = LRNModel.convolution(relu3, c4filter, [1, 1])  # (6, 6, 128)
relu4 = LRNModel.relu(conv4)
# layer5 卷积 ReLU maxpooling
c5filter = Parameter.kernel([3, 3, 128, 256], 0.1)
conv5 = LRNModel.convolution(relu4, c5filter, [1, 1])  # (4, 4, 256)
relu5 = LRNModel.relu(conv5)
maxpool5 = LRNModel.max_pooling(relu5, [2, 2], [1, 1])  # (3, 3, 256)
# flatten
m5_shape = maxpool5.shape
flatten_size = [-1, m5_shape[1] * m5_shape[2]
                * m5_shape[3]]
flatten_input = LRNModel.flatten(maxpool5, flatten_size)  # (batch_size, 3 * 3 * 256)
# layer1 Full-connected  ReLU dropout
fc1_w = Parameter.kernel([m5_shape[1] * m5_shape[2] * m5_shape[3], 1024], 1e-2)
fc1_b = Parameter.bias([1024], 1.0)
fc1 = LRNModel.full_connect(flatten_input, fc1_w, fc1_b)
relufc1 = LRNModel.relu(fc1)
dropout1 = LRNModel.dropout(relufc1, 0.1)
# layer2 Full-connected ReLU dropout
fc2_w = Parameter.kernel([1024, 512], 1e-2)
fc2_b = Parameter.bias([512], 1.0)
fc2 = LRNModel.full_connect(dropout1, fc2_w, fc2_b)
relufc2 = LRNModel.relu(fc2)
dropout2 = LRNModel.dropout(relufc2, 0.1)
# layer3 Full-connected softmax
fc3_w = Parameter.kernel([512, 10], 1e-2)
fc3_b = Parameter.bias([10], 1.0)
fc3 = LRNModel.full_connect(dropout2, fc3_w, fc3_b)
# softmax
hx = tf.nn.softmax(fc3)
# optimizer
loss, optimizer = Parameter.cost_and_optimizer(y, hx, 1e-4)
# accuracy
accuracy = Parameter.pred_and_accuracy(y, hx)

# train
epoch = 10
batch_size = 120
test_batch = 100
train_size = train_images.shape[0]
test_size = test_images.shape[0]
print(train_labels.shape)
with tf.compat.v1.Session() as sess:
    global_variables = tf.compat.v1.global_variables_initializer()
    sess.run(global_variables)
    for e in range(epoch):
        iteration = 0
        epoch_accuracy, epoch_loss = [], []
        while batch_size * iteration < train_size:
            batch_start = iteration * batch_size
            batch_end = min(train_size, batch_size * (iteration + 1))
            batch_x, batch_y = \
                train_images[batch_start: batch_end], \
                train_labels[batch_start: batch_end]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            accuracy_rate = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            epoch_accuracy.append(accuracy_rate)
            loss_rate = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            epoch_loss.append(loss_rate)
            # output accuracy when iteration % 30 == 0 or going to end this epoch
            if iteration % 30 == 0 or batch_end == train_size:
                print('training: {} / {}, accuracy: {}, loss: {}'
                      .format(batch_end, train_size,
                              np.mean(epoch_accuracy), np.mean(epoch_loss)))
            iteration = iteration + 1
        print('epoch: {}, accuracy: {}, loss: {}'
              .format(e + 1, np.mean(epoch_accuracy), np.mean(epoch_loss)))
    # test
    print('\n\nrunning test:')
    iteration_test = int(test_size / test_batch)
    test_acc, test_loss = [], []
    for i in range(iteration_test):
        batch_start = i * test_batch
        batch_end = min(train_size, test_batch * (i + 1))
        batch_test_x, batch_test_y = \
            test_images[batch_start: batch_end], \
            test_labels[batch_start: batch_end]
        test_acc.append(
            sess.run(accuracy, feed_dict={x: batch_test_x, y: batch_test_y}))
        test_loss.append(
            sess.run(loss, feed_dict={x: batch_test_x, y: batch_test_y}))
        if i % 20 == 0:
            print('test_size: {}, test accuracy: {}, test loss: {}'
                  .format(batch_end, np.mean(test_acc), np.mean(test_loss)))
    print('test accuracy: {}, test loss: {}'
          .format(np.mean(test_acc), np.mean(test_loss)))
