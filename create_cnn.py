import tensorflow as tf


class Create_CNN:

    def _conv(self, input, weights, biases, kernelStride):
        conv = tf.nn.conv2d(input, weights, strides=[1, kernelStride, kernelStride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv)
        return conv

    def create_cnn(self, input, nChannels, kernelSize, kernelStride):
        weight_two = tf.convert_to_tensor(tf.random_uniform([kernelSize, kernelSize, nChannels, nChannels],
                                                            minval=-0.08, maxval=0.08))
        bias = tf.convert_to_tensor(tf.random_uniform([30], maxval=1.0, minval=0.0))
        # x = tf.reshape(input, shape=[-1, 100, 106, 3])
        layer1 = Create_CNN()._conv(input, weight_two, bias, kernelStride)
        layer2 = Create_CNN()._conv(layer1, weight_two, bias, kernelStride)
        layer3 = Create_CNN()._conv(layer2, weight_two, bias, kernelStride)
        layer4 = Create_CNN()._conv(layer3, weight_two, bias, kernelStride)
        layer5 = Create_CNN()._conv(layer4, weight_two, bias, kernelStride)
        return layer5

    def create_FCNN(self, input, nChannels, kernelSize, kernelStride):
        weight_one = tf.convert_to_tensor(tf.random_uniform([kernelSize, kernelSize, 3, nChannels], minval=-0.08, maxval=0.08))
        bias = tf.convert_to_tensor(tf.random_uniform([30], maxval=1.0, minval=0.0))
        x = tf.reshape(input, shape=(-1, 500, 530, 3))
        layer = Create_CNN()._conv(x, weight_one, bias, kernelStride)
        return layer

    def create_SingCNN(self, input, nChannels, kernelSize, kernelStride):
        weight_one = tf.convert_to_tensor(tf.random_uniform([kernelSize, kernelSize, nChannels, nChannels], minval=-0.08, maxval=0.08))
        bias = tf.convert_to_tensor(tf.random_uniform([30], maxval=1.0, minval=0.0))
        layer = Create_CNN()._conv(input, weight_one, bias, kernelStride)
        return layer



