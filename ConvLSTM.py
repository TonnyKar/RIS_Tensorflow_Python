import tensorflow as tf
from create_cnn import Create_CNN


class ConvLSTM:

    def convlstm(self, xt, ht, ct):
        bias = tf.convert_to_tensor(tf.random_uniform([30], maxval=1.0, minval=0.0))
        i_t = tf.sigmoid(tf.nn.bias_add(Create_CNN().create_SingCNN(input=xt, nChannels=30, kernelSize=3, kernelStride=1) +
                         Create_CNN().create_SingCNN(input=ht, nChannels=30, kernelSize=3, kernelStride=1), bias))

        bias = tf.convert_to_tensor(tf.random_uniform([30], maxval=1.0, minval=0.0))
        f_t = tf.sigmoid(
            tf.nn.bias_add(Create_CNN().create_SingCNN(input=xt, nChannels=30, kernelSize=3, kernelStride=1) +
                           Create_CNN().create_SingCNN(input=ht, nChannels=30, kernelSize=3, kernelStride=1), bias))

        bias = tf.convert_to_tensor(tf.random_uniform([30], maxval=1.0, minval=0.0))
        o_t = tf.sigmoid(
            tf.nn.bias_add(Create_CNN().create_SingCNN(input=xt, nChannels=30, kernelSize=3, kernelStride=1) +
                           Create_CNN().create_SingCNN(input=ht, nChannels=30, kernelSize=3, kernelStride=1), bias))

        bias = tf.convert_to_tensor(tf.random_uniform([30], maxval=1.0, minval=0.0))
        g_t = tf.tanh(
            tf.nn.bias_add(Create_CNN().create_SingCNN(input=xt, nChannels=30, kernelSize=3, kernelStride=1) +
                           Create_CNN().create_SingCNN(input=ht, nChannels=30, kernelSize=3, kernelStride=1), bias))

        c_t = tf.multiply(f_t, ct) + tf.multiply(i_t, g_t)
        h_t = tf.multiply(o_t, tf.tanh(c_t))

        return c_t, h_t