from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from AlexNet_BN import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.model_selection import train_test_split
# parameter
train_epoch = 100
train_batch_size = 500
(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.cifar10.load_data()
train_labels = train_labels.reshape(train_labels.shape[0])
test_labels = test_labels.reshape(test_labels.shape[0])
# To prevent multiplying large number, we need to rescale.
train_images = train_images.astype(np.float)
test_images = test_images.astype(np.float)
train_images /= 255.0  # rescale
test_images /= 255.0  # rescale
# train and validation split
train_images, val_images, train_labels, val_labels = \
    train_test_split(train_images, train_labels,
                     test_size=0.1, stratify=train_labels)
# one-hot
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
val_labels = tf.keras.utils.to_categorical(val_labels, 10)
print(train_images.shape, train_labels.shape,
      test_images.shape, test_labels.shape,
      val_images.shape, val_labels.shape)
# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

# Batch Normalization model initialization
CIFAR10_BN_Model = Model((32, 32, 3), (32, 32, 3))
# layer1 convolution ReLU BN maxpooling
CIFAR10_BN_Model.conv2D(64, (3, 3), (1, 1))  # (30, 30, 64)
CIFAR10_BN_Model.batch_normalization(3)
CIFAR10_BN_Model.max_pooling((4, 4), (2, 2))  # (14, 14, 64)
# layer2 convolution ReLU BN maxpooling
CIFAR10_BN_Model.conv2D(128, (3, 3), (1, 1))  # (12, 12, 128)
CIFAR10_BN_Model.batch_normalization(3)
CIFAR10_BN_Model.max_pooling((3, 3), (1, 1))  # (10, 10, 128)
# layer3 convolution ReLU
CIFAR10_BN_Model.conv2D(256, (3, 3), (1, 1))  # (8, 8, 256)
# layer4 convolution ReLU
CIFAR10_BN_Model.conv2D(256, (2, 2), (1, 1))  # (7, 7, 256)
# layer5 convolution ReLU maxpooling
CIFAR10_BN_Model.conv2D(380, (2, 2), (1, 1))  # (6, 6, 380)
CIFAR10_BN_Model.max_pooling((3, 3), (1, 1))  # (4, 4, 380)
# Flatten
CIFAR10_BN_Model.flatten()
# FC1 ReLU Dropout
CIFAR10_BN_Model.full_connected_layer(1024, 'relu')
CIFAR10_BN_Model.dropout(0.5)
# FC2 ReLU Dropout
CIFAR10_BN_Model.full_connected_layer(256, 'relu')
CIFAR10_BN_Model.dropout(0.5)
# FC3 softmax
CIFAR10_BN_Model.full_connected_layer(10, 'softmax')
# show model
CIFAR10_BN_Model.summary()
# Generator
train_aug = ImageDataGenerator(width_shift_range=0.1,
                               height_shift_range=0.1,
                               horizontal_flip=True)
train_aug.fit(train_images)
train_generator = train_aug.flow(train_images, train_labels, batch_size=train_batch_size)
# prevent overfitting
es = tf.keras.callbacks.EarlyStopping('val_loss', patience=25,
                                      restore_best_weights=True)
mc = tf.keras.callbacks.ModelCheckpoint('AlexNet_BN.h5', monitor='val_loss',
                                        save_best_only=True,
                                        mode='min')
cblist = [es, mc]
# train model
CIFAR10_BN_Model.train(train_images, train_labels,
                       train_batch_size, train_epoch,
                       generator=train_generator, one_hot=True, cblist=cblist,
                       val_images=val_images, val_labels=val_labels)
# test and evaluate the model for last step
test_loss, test_acc = CIFAR10_BN_Model.train_error(test_images,
                                                   test_labels)
print('best step:\ntest loss: {}, test accuracy: {}'.format(test_loss, test_acc))
