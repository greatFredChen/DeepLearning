'''
Based on tensorflow 2.1.0
by Fred Chen
'''
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Reshape, Dense, BatchNormalization, Flatten
from tensorflow.keras.optimizers import SGD, Nadam, Adam


class Model:
    def __init__(self, model_size, input_shape):
        '''

        :param model_size: [height, width, channel]
        :param input_shape: [height, width]
        '''
        self.model = Sequential()
        self.model.add(Reshape(model_size, input_shape=input_shape))

    def conv2D(self, filter, kernel_size, strides):
        '''

        :param filter: the dimensionality of the output space
        :param kernel_size: the height and width of the 2D convolution window
        :param strides: the strides of the convolution along the height and width
        :return:
        '''
        self.model.add(Conv2D(filter, kernel_size, activation='relu',
                              strides=strides, padding='valid'))  # 只卷积有效部分，不补全padding

    def max_pooling(self, pool_size, strides):
        '''

        :param pool_size: factors by which to downscale, just like convolution window
        :param strides: the strides of the maxpool along the height and width
        :return:
        '''
        self.model.add(MaxPool2D(pool_size, strides))

    def full_connected_layer(self, units, activation):
        '''

        :param units: The number of neurons in this layer
        :param activation: activation function such as relu, sigmoid, softmax, etc.
        :return:
        '''
        self.model.add(Dense(units, activation=activation))

    def batch_normalization(self, axis=3):
        '''

        :param axis: batch normalization based on channel, axis --> channel
        :return:
        '''
        self.model.add(BatchNormalization(axis))  # 默认gamma和beta

    def flatten(self):
        '''

        :return:
        '''
        self.model.add(Flatten(data_format='channels_last'))

    # dropout降低过拟合
    def dropout(self, rate):
        '''

        :param rate: Fraction of the input units to drop.
        :return:
        '''
        self.model.add(Dropout(rate=rate))

    def summary(self):
        '''

        :return:
        '''
        return self.model.summary()

    def train(self, train_images, train_labels, batch_size, epochs, **kwargs):
        '''

        :param train_images: data of training images
        :param train_labels: data of training labels
        :param batch_size: The number of samples we train in an iteration
        :param epochs: we train all the samples in the training set once an epoch.
        :param generator: ImageDataGenerator using for data augmentation
        :param one_hot: train_labels use one-hot or not.
        :return:
        '''
        generator = kwargs.get('generator')
        one_hot = kwargs.get('one_hot')
        cblist = kwargs.get('cblist')
        val_images = kwargs.get('val_images')
        val_labels = kwargs.get('val_labels')
        # 使用adam法优化，损失函数使用交叉熵，并输出accuracy
        optimizer = SGD(0.01, 0.9)
        if not one_hot:
            self.model.compile(optimizer=optimizer,
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            self.model.compile(optimizer=optimizer,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        if not generator:
            self.model.fit(train_images, train_labels,
                           batch_size=batch_size, epochs=epochs,
                           callbacks=cblist,
                           validation_data=(val_images, val_labels))
        else:
            self.model.fit(generator, steps_per_epoch=len(train_images) / batch_size,
                           epochs=epochs, callbacks=cblist,
                           validation_data=(val_images, val_labels))

    def train_error(self, test_images, test_labels):
        '''

        :param test_images: data of test images
        :param test_labels: data of test labels
        :param generator: ImageDataGenerator using for testing
        :return:
        '''
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        return test_loss, test_acc
