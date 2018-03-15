#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:45:36 2018

@author: Daniel Salazar
"""
# Minh Nguyen, Pradeep Ravi, Stuti Patel, Vijaykumar Mamidala, and Daniel Salazar
import cv2,sys,argparse,os,glob,time,math,random

from datetime import timedelta
from sklearn.utils import shuffle

import numpy as np
from numpy.random import seed

import tensorflow as tf
from tensorflow import set_random_seed

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

class DataSet(object):

    def __init__(
        self,
        images,
        labels,
        img_names,
        cls,
        ):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:

      # After each epoch we update this

            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return (self._images[start:end], self._labels[start:end],
                self._img_names[start:end], self._cls[start:end])


# Prepdata
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*m')
        #path = os.path.join(train_path, fields + "/complex", '*m')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images, labels, img_names, cls


def read_train_sets(
    train_path,
    image_size,
    classes,
    validation_size,
    ):

    class DataSets(object):

        pass

    data_sets = DataSets()

    (images, labels, img_names, cls) = load_train(train_path, image_size, classes)
    (images, labels, img_names, cls) = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels,
                              train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels,
                              validation_img_names, validation_cls)

    return data_sets


def imgGen(
    img,
    zca=False,
    rotation=0.,
    w_shift=0.,
    h_shift=0.,
    shear=0.,
    zoom=0.,
    h_flip=False,
    v_flip=False,
    preprocess_fcn=None,
    batch_size=9,
    ):
    datagen = ImageDataGenerator(
        zca_whitening=zca,
        rotation_range=rotation,
        width_shift_range=w_shift,
        height_shift_range=h_shift,
        shear_range=shear,
        zoom_range=zoom,
        fill_mode='nearest',
        horizontal_flip=h_flip,
        vertical_flip=v_flip,
        preprocessing_function=preprocess_fcn,
        data_format=K.image_data_format(),
        )

    datagen.fit(img)


# CalculateAccuracy

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(
    input,
    num_input_channels,
    conv_filter_size,
    num_filters,
    ):

    # # We shall define the weights that will be trained using create_weights function.

    weights = create_weights(shape=[conv_filter_size, conv_filter_size,
                             num_input_channels, num_filters])

    # # We create biases using the create_biases function. These are also trained.

    biases = create_biases(num_filters)

    # # Creating the convolutional layer

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1,
                         1], padding='SAME')

    layer += biases

    # # We shall be using max-pooling.

    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1,
                           2, 2, 1], padding='SAME')

    # # Output of pooling is fed to Relu which is the activation function for us.

    layer = tf.nn.relu(layer)
    return layer


def create_flatten_layer(layer):

    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.

    layer_shape = layer.get_shape()

    # # Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.

    num_features = layer_shape[1:4].num_elements()

    # # Now, we Flatten the layer so we shall have to reshape to num_features

    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(
    input,
    num_inputs,
    num_outputs,
    use_relu=True,
    ):

    # Let's define trainable weights and biases.

    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def show_progress(
    epoch,
    feed_dict_train,
    feed_dict_validate,
    val_loss,
    ):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = \
        'Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}'
    print msg.format(epoch + 1, acc, val_acc, val_loss)


def train(num_iteration):
    global total_iterations

    for i in range(total_iterations, total_iterations + num_iteration):
        (x_batch, y_true_batch, _, cls_batch) = data.train.next_batch(batch_size)
        (x_valid_batch, y_valid_batch, _, valid_cls_batch) = data.valid.next_batch(batch_size)
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './hand_gestures_model')
    total_iterations += num_iteration



"""
This file is where we construct and train the model
"""

seed(1)
set_random_seed(2)
batch_size = 32

# Prepare input data

classes = [
    'A',
    'B',
    'C',
    'Five',
    'Point',
    'V',
    ]
num_classes = len(classes)

# 20% of the data will automatically be used for validation

validation_size = 0.1
img_size = 100
num_channels = 3
train_path = os.path.join(os.getcwd(), 'Marcel-Train')

# We shall load all the training and validation images and labels into memory using openCV and use that during training

data = read_train_sets(train_path, img_size, classes,
                       validation_size=validation_size)

print 'Complete reading input data. Will Now print a snippet of it'
print 'Number of files in Training-set:\t\t{}'.format(len(data.train.labels))
print 'Number of files in Validation-set:\t{}'.format(len(data.valid.labels))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size,
                   num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes],
                        name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

##Network graph params
# filter_size_conv1 = 3
# num_filters_conv1 = 32
#
# filter_size_conv2 = 3
# num_filters_conv2 = 32
#
# filter_size_conv3 = 3
# num_filters_conv3 = 64
filter_size_conv1 = 3
num_filters_conv1 = 16

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

filter_size_conv4 = 5
num_filters_conv4 = 128

fc_layer_size = 80

layer_conv1 = create_convolutional_layer(input=x,
        num_input_channels=num_channels,
        conv_filter_size=filter_size_conv1,
        num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
        num_input_channels=num_filters_conv1,
        conv_filter_size=filter_size_conv2,
        num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
        num_input_channels=num_filters_conv2,
        conv_filter_size=filter_size_conv3,
        num_filters=num_filters_conv3)

#
# layer_conv4 = create_convolutional_layer(input=layer_conv3,
#                                          num_input_channels=num_filters_conv3,
#                                          conv_filter_size=filter_size_conv4,
#                                          num_filters=num_filters_conv4)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size, use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size,
                            num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = \
    tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer())
total_iterations = 0
saver = tf.train.Saver()
train(num_iteration=10000)



# First, pass the path of the image

"""
This file is intended to use for live presentation.
When we pass a image of a gesture in, we get probabilities it belongs to each class
"""

dir_path = os.path.dirname(os.path.realpath('__file__'))
image_path = 'Marcel-Test/Point/uniform/Point-uniform19.ppm'
filename = dir_path + '/' + image_path
image_size = 100
num_channels = 3
images = []

# Reading the image using OpenCV

image = cv2.imread(filename)

# Resizing the image to our desired size and preprocessing will be done exactly as done during training

image = cv2.resize(image, (image_size, image_size), 0, 0,
                   cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0 / 255.0)

# The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.

x_batch = images.reshape(1, image_size, image_size, num_channels)

## Let us restore the saved model

sess = tf.Session()

# Step-1: Recreate the network graph. At this step only graph is created.

saver = tf.train.import_meta_graph('hand_gestures_model.meta')

# Step-2: Now let's load the weights saved using the restore method.

saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored

graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network

y_pred = graph.get_tensor_by_name('y_pred:0')

## Let's feed the images to the input placeholders

x = graph.get_tensor_by_name('x:0')
y_true = graph.get_tensor_by_name('y_true:0')
y_test_images = np.zeros((1, 6))

### Creating the feed_dict that is required to be fed to calculate y_pred

feed_dict_testing = {x: x_batch, y_true: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)

# result is of this format [probabiliy_of_rose probability_of_sunflower]

print result


# Prepare input data

"""
Code to calculate the average accuracy rate when applying the trained model on testing data is stored here
"""
def load_test(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        #path = os.path.join(train_path, fields, '*m')
        path = os.path.join(train_path, fields + "/uniform", '*m')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images, labels, img_names, cls

classes = [
    'A',
    'B',
    'C',
    'Five',
    'Point',
    'V',
    ]
num_classes = len(classes)

img_size = 100
num_channels = 3
train_path = 'Marcel-Test'

testing_data = load_test(train_path, img_size, classes)

x_batch = testing_data[0]

## Let us restore the saved model

sess = tf.Session()

# Step-1: Recreate the network graph. At this step only graph is created.

saver = tf.train.import_meta_graph('hand_gestures_model.meta')

# Step-2: Now let's load the weights saved using the restore method.

saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored

graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network

y_pred = graph.get_tensor_by_name('y_pred:0')

## Let's feed the images to the input placeholders

x = graph.get_tensor_by_name('x:0')
y_true = graph.get_tensor_by_name('y_true:0')

y_test_images = np.zeros((len(testing_data[0]), 6))

### Creating the feed_dict that is required to be fed to calculate y_pred

feed_dict_testing = {x: x_batch, y_true: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)
y_predicted = tf.argmax(result, dimension=1)
y_true = tf.argmax(testing_data[1], dimension=1)
correct_prediction = tf.equal(y_predicted, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = sess.run(accuracy)
print acc
temp = 0

"""
0.486911
"""
