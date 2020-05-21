from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from AlexNet_BN import Model
import numpy as np
import os
train_epoch = 100
train_batch_size = 500
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.astype(np.float)
test_images = test_images.astype(np.float)
train_images /= 255.0
test_images /= 255.0
print(train_images.shape, test_images.shape)
# print(train_images.shape)
# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

# BN model part
# 初始化模型
AlexNet_model_BN = Model((28, 28, 1), (28, 28))
# layer1 Convolution 输入 卷积 ReLU BN maxpooling
AlexNet_model_BN.conv2D(8, (3, 3), (1, 1))  # (26, 26, 8)
AlexNet_model_BN.batch_normalization(3)
AlexNet_model_BN.max_pooling((4, 4), (2, 2))  # (12, 12, 8)
# layer2 Convolution 卷积 ReLU BN maxpooling
AlexNet_model_BN.conv2D(16, (3, 3), (1, 1))  # (10, 10, 16)
AlexNet_model_BN.batch_normalization(3)
AlexNet_model_BN.max_pooling((3, 3), (1, 1))  # (8, 8, 16)
# layer3 Convolution 卷积 ReLU
AlexNet_model_BN.conv2D(32, (2, 2), (1, 1))  # (7, 7, 32)
# layer4 Convolution 卷积 ReLU
AlexNet_model_BN.conv2D(32, (2, 2), (1, 1))  # (6, 6, 32)
# layer5 Convolution 卷积 ReLU maxpooling
AlexNet_model_BN.conv2D(64, (3, 3), (1, 1))  # (4, 4, 64)
AlexNet_model_BN.max_pooling((2, 2), (1, 1))  # (3, 3, 64)
# Flatten
AlexNet_model_BN.flatten()  # (batch_size, 3 * 3 * 64)
# layer1 Full-Connected Network ReLU dropout
AlexNet_model_BN.full_connected_layer(64, 'relu')
AlexNet_model_BN.dropout(0.1)  # drop 10% units randomly
# layer2 Full-Connected Network ReLU dropout
AlexNet_model_BN.full_connected_layer(32, 'relu')
AlexNet_model_BN.dropout(0.1)  # drop 10% units randomly
# layer3 Full-Connected Network softmax
AlexNet_model_BN.full_connected_layer(10, 'softmax')
# show model
AlexNet_model_BN.summary()
# prevent overfitting
es = tf.keras.callbacks.EarlyStopping('val_loss', patience=train_epoch / 10,
                                      restore_best_weights=True)
# train model
AlexNet_model_BN.train(train_images, train_labels, train_batch_size,
                       train_epoch, one_hot=False,
                       test_images=test_images,
                       test_labels=test_labels, es=es)
h = AlexNet_model_BN.get_history()
# test error
test_loss, test_acc = AlexNet_model_BN.train_error(test_images, test_labels)
print('last step:\ntest loss: {}, test accuracy: {}'.format(test_loss, test_acc))
# test and evaluate the model for last step for best step
print('best step:\ntest loss: {}, test accuracy: {}'.
      format(np.min(h.history['val_loss']),
             np.max(h.history['val_accuracy'])))
