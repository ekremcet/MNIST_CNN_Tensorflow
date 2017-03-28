from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.ndimage import rotate
from scipy.ndimage import zoom
from scipy.ndimage import shift
from random import randint

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

class ekrem_cetinkaya_training:

    def __init__(self,network_type,dataset_type,MNIST_path):
        self.network_type = network_type
        self.dataset_type = dataset_type
        self.MNIST_path = MNIST_path

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,input, filter):
        return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,value):
        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def down_sample(self,input, output):
        newPixel = 0
        for imageIndex in range(len(input) - 1):
            # Get each image
            image = input[imageIndex]
            orjPixel = 0
            # Get the Average value of 2x2 and assign it to the output
            while orjPixel < 728:
                if (orjPixel % 28) - 2 == 0:
                    orjPixel += 30
                val = (image[orjPixel] + image[orjPixel + 1] + image[orjPixel + 28] + image[orjPixel + 29]) / 4
                orjPixel += 2
                newPixel += 1
                output[imageIndex][newPixel] = val
            newPixel = 0

        return output

    def augment_image(self,input, output):
        for index in range(len(input) - 1):
            output[index] = input[index]
        # first augmentation - Rotate images to right / Italic
        for imageIndex in range(len(input) - 1):
            image = input[imageIndex]
            newImg = np.reshape(image, [14, 14])
            augmented_image = rotate(newImg, 10, reshape=False)
            final_image = np.reshape(augmented_image, [196])
            output[imageIndex + 9000] = final_image
        # Second augmentation Rotate images randomly
        for imageIndex in range(len(input) - 1):
            degree = randint(-15,15)
            image = input[imageIndex]
            newImg = np.reshape(image, [14, 14])
            augmented_image = rotate(newImg, degree, reshape=False)
            final_image = np.reshape(augmented_image, [196])
            output[imageIndex + 18000] = final_image
        # Final augmentation - Shifting images
        for imageIndex in range(len(input) - 1):
            random = randint(-3,3)
            shift_amount = 0.1 * random
            image = input[imageIndex]
            newImg = np.reshape(image, [14, 14])
            augmented_image = shift(newImg,shift_amount)
            final_image = np.reshape(augmented_image, [196])
            output[imageIndex + 27000] = final_image

        return output

    def network1(self):
        # Import data
        mnist = input_data.read_data_sets(self.MNIST_path, one_hot=True)
        place_holder = 196
        image_reshape = 14
        flat_reshape = 4
        max_step = 180
        if self.dataset_type == "28x28_dataset":
            place_holder = 784
            flat_reshape = 7
            image_reshape = 28
        elif self.dataset_type == "14x14_augmented_dataset":
            max_step = 720

        # Input Layer
        inputImage = tf.placeholder(tf.float32, [None, place_holder])
        inputLabel = tf.placeholder(tf.float32, [None, 10])
        sess = tf.InteractiveSession()

        inputReshaped = tf.reshape(inputImage, [-1, image_reshape, image_reshape, 1])

        # Layer 1-2-3 - First conv + maxp + relu
        Weight_conv1 = self.weight_variable([5, 5, 1, 32])  # 5x5 convolution, 1 channel, 32 feature
        bias_conv1 = self.bias_variable([32])

        h_conv1 = tf.nn.relu(self.conv2d(inputReshaped, Weight_conv1) + bias_conv1)  # Convolution + Relu
        h_pool1 = self.max_pool_2x2(h_conv1)

        # Layer 4-5-6 - Second conv + maxp + relu
        Weight_conv2 = self.weight_variable([5, 5, 32, 64])  # 5x5 Conv, 32 channel, 64 feature
        bias_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, Weight_conv2) + bias_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # Layer 7 - FullyConnected 1
        h_pool2_flat = tf.reshape(h_pool2, [-1, flat_reshape * flat_reshape * 64])

        Weight_fullyCon1 = self.weight_variable([flat_reshape * flat_reshape * 64, 1024])
        bias_fullyCon1 = self.bias_variable([1024])

        h_fullyCon1 = tf.nn.relu(tf.matmul(h_pool2_flat, Weight_fullyCon1) + bias_fullyCon1)

        # Layer 8 - Dropout + FullyConnected 2 Output layer
        keep_prob = tf.placeholder(tf.float32)
        h_fullyCon1_drop = tf.nn.dropout(h_fullyCon1, keep_prob)

        Weight_fullyCon2 = self.weight_variable([1024, 10])
        bias_fullyCon2 = self.bias_variable([10])

        predictedClass = tf.matmul(h_fullyCon1_drop, Weight_fullyCon2) + bias_fullyCon2

        # Evaluation
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=inputLabel, logits=predictedClass))

        correct_prediction = tf.equal(tf.argmax(predictedClass, 1), tf.argmax(inputLabel, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Training Algoritmasi
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        imageTrain = mnist.test.images[0:9000]
        labelTrain = mnist.test.labels[0:9000]

        imageTest = mnist.test.images[9000:10000]
        labelTest = mnist.test.labels[9000:10000]

        if self.dataset_type == "14x14_dataset":
            temp_image = mnist.test.images[0:9000]
            imageTrain = self.down_sample(temp_image,np.zeros(shape=(9000,196)))

            temp_tests = mnist.test.images[9000:10000]
            imageTest = self.down_sample(temp_tests,np.zeros(shape=(1000,196)))

        elif self.dataset_type == "14x14_augmented_dataset":
            temp_image = mnist.test.images[0:9000]
            down_sampled_images = self.down_sample(temp_image,np.zeros(shape=(9000,196)))
            imageTrain = self.augment_image(down_sampled_images,np.zeros(shape=(36000,196)))

            labelTrain = np.zeros(shape=(36000, 10))
            labelTrain[0:9000] = mnist.test.labels[0:9000]
            labelTrain[9000:18000] = mnist.test.labels[0:9000]
            labelTrain[18000:27000] = mnist.test.labels[0:9000]
            labelTrain[27000:36000] = mnist.test.labels[0:9000]

            temp_tests = mnist.test.images[9000:10000]
            imageTest = self.down_sample(temp_tests, np.zeros(shape=(1000, 196)))

        # Training_steps
        sess.run(tf.global_variables_initializer())

        for step in range(max_step):
            batch_inputImage = imageTrain[50 * (step % 9000):50 * (step % 9000) + 50]
            batch_inputLabel = labelTrain[50 * (step % 9000):50 * (step % 9000) + 50]
            if step % 10 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={inputImage: batch_inputImage, inputLabel: batch_inputLabel, keep_prob: 1.0})
                print("step %d:, training accuarcy %g" % (step, train_accuracy))
            train_step.run(feed_dict={inputImage: batch_inputImage, inputLabel: batch_inputLabel, keep_prob: 0.5})
        # Result
        print("test accuracy %g" % accuracy.eval(
            feed_dict={inputImage: imageTest, inputLabel: labelTest, keep_prob: 1.0}))

    def network2(self):
        # Import data
        mnist = input_data.read_data_sets(self.MNIST_path, one_hot=True)
        place_holder = 196
        image_reshape = 14
        flat_reshape = 4
        max_step = 180
        if self.dataset_type == "28x28_dataset":
            place_holder = 784
            flat_reshape = 7
            image_reshape = 28
        elif self.dataset_type == "14x14_augmented_dataset":
            max_step = 720

        # Input Layer
        inputImage = tf.placeholder(tf.float32, [None, place_holder])
        inputLabel = tf.placeholder(tf.float32, [None, 10])
        sess = tf.InteractiveSession()

        inputReshaped = tf.reshape(inputImage, [-1, image_reshape, image_reshape, 1])

        # Layer 1-2-3 - First conv + maxp + relu
        Weight_conv1 = self.weight_variable([3, 3, 1, 64])  # 3x3 convolution, 64 feature
        bias_conv1 = self.bias_variable([64])

        h_conv1 = tf.nn.relu(self.conv2d(inputReshaped, Weight_conv1) + bias_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # Layer 4-5-6 - Second conv + maxp + relu
        Weight_conv2 = self.weight_variable([3, 3, 64, 128])  # 3x3 Conv, 8 channel, 128 feature
        bias_conv2 = self.bias_variable([128])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, Weight_conv2) + bias_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # Layer 7 - FullyConnected 1
        h_pool2_flat = tf.reshape(h_pool2, [-1, flat_reshape * flat_reshape * 128])

        Weight_fullyCon1 = self.weight_variable([flat_reshape * flat_reshape * 128, 4096])
        bias_fullyCon1 = self.bias_variable([4096])

        h_fullyCon1 = tf.nn.relu(tf.matmul(h_pool2_flat, Weight_fullyCon1) + bias_fullyCon1)

        # Layer 8 - Dropout + FullyConnected 2 Output layer
        keep_prob = tf.placeholder(tf.float32)
        h_fullyCon1_drop = tf.nn.dropout(h_fullyCon1, keep_prob)

        Weight_fullyCon2 = self.weight_variable([4096, 10])
        bias_fullyCon2 = self.bias_variable([10])

        predictedClass = tf.matmul(h_fullyCon1_drop, Weight_fullyCon2) + bias_fullyCon2

        # Evaluation
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=inputLabel, logits=predictedClass))

        correct_prediction = tf.equal(tf.argmax(predictedClass, 1), tf.argmax(inputLabel, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Training Algoritmasi
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        imageTrain = mnist.test.images[0:9000]
        labelTrain = mnist.test.labels[0:9000]

        imageTest = mnist.test.images[9000:10000]
        labelTest = mnist.test.labels[9000:10000]

        if self.dataset_type == "14x14_dataset":
            temp_image = mnist.test.images[0:9000]
            imageTrain = self.down_sample(temp_image, np.zeros(shape=(9000, 196)))

            temp_tests = mnist.test.images[9000:10000]
            imageTest = self.down_sample(temp_tests, np.zeros(shape=(1000, 196)))

        elif self.dataset_type == "14x14_augmented_dataset":
            temp_image = mnist.test.images[0:9000]
            down_sampled_images = self.down_sample(temp_image, np.zeros(shape=(9000, 196)))
            imageTrain = self.augment_image(down_sampled_images, np.zeros(shape=(36000, 196)))

            labelTrain = np.zeros(shape=(36000, 10))
            labelTrain[0:9000] = mnist.test.labels[0:9000]
            labelTrain[9000:18000] = mnist.test.labels[0:9000]
            labelTrain[18000:27000] = mnist.test.labels[0:9000]
            labelTrain[27000:36000] = mnist.test.labels[0:9000]

            temp_tests = mnist.test.images[9000:10000]
            imageTest = self.down_sample(temp_tests, np.zeros(shape=(1000, 196)))

        # Training_steps
        sess.run(tf.global_variables_initializer())

        for step in range(max_step):
            batch_inputImage = imageTrain[50 * (step % 9000):50 * (step % 9000) + 50]
            batch_inputLabel = labelTrain[50 * (step % 9000):50 * (step % 9000) + 50]
            if step % 10 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={inputImage: batch_inputImage, inputLabel: batch_inputLabel, keep_prob: 1.0})
                print("step %d:, training accuarcy %g" % (step, train_accuracy))
            train_step.run(feed_dict={inputImage: batch_inputImage, inputLabel: batch_inputLabel, keep_prob: 0.5})
        # Result
        print("test accuracy %g" % accuracy.eval(
            feed_dict={inputImage: imageTest, inputLabel: labelTest, keep_prob: 1.0}))

    def run_training(self):
        if self.network_type == "network1":
            self.network1()
        elif self.network_type == "network2":
            self.network2()

if __name__ == '__main__':
    training = ekrem_cetinkaya_training("network2","14x14_augmented_dataset","/tmp/tensorflow/mnist/input_data")
    training.run_training()