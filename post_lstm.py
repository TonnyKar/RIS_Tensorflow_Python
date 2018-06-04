import tensorflow as tf
from create_cnn import Create_CNN


class Post_LSTM:
    def dictionary(self, tranined_input, upsamplingScale):
        kernelSize = 1
        kernelStride = 1
        nChannels = 30
        #conv = Create_CNN().remove_Filter(input=tranined_input, nChannels=nChanells, kernelSize=filterSize, kernelStride=kernelStride)

        filter = tf.random_normal([kernelSize, kernelSize, nChannels, 1])
        layer = tf.nn.conv2d(input=tranined_input, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.reshape(layer, shape=[1, 100, 106])
        bias = tf.convert_to_tensor(tf.random_uniform([106], maxval=1.0, minval=0.0))
        log_softmax = tf.nn.log_softmax(conv)
        biased_log = tf.nn.bias_add(log_softmax, bias)
        sigmoid_log = tf.sigmoid(biased_log)
        #upsamplingImage = self.UpSampling(input=sigmoid_log, upsamplingScale=upsamplingScale)
        reshape_dict = tf.reshape(sigmoid_log, shape=[100 * 106])
        return reshape_dict

    def scores(self, trained_input, ksize):
        maxpooling = tf.nn.max_pool(trained_input, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1],
                                    padding='SAME')
        linear_data = self.LinearTransformation(trained_input)
        linear_data = tf.sigmoid(linear_data)
        return linear_data

    def UpSampling(self, input, upsamplingScale):
        old_width = input.shape[1]
        old_height = input.shape[2]
        new_width = old_width * upsamplingScale
        new_height = old_height * upsamplingScale
        resized = tf.image.resize_images(input, [new_width, new_height], method=tf.image.ResizeMethod.BILINEAR)
        return resized

    def LinearTransformation(self, trainedInput):
        size = tf.size(trainedInput)
        trained_input_flat = tf.reshape(trainedInput, [-1, size])
        weights = tf.random_uniform([size, 1], minval=-0.08, maxval=0.08)
        bias = tf.convert_to_tensor(10, dtype=tf.float32)
        LinearTransform = tf.add(tf.matmul(trained_input_flat, weights), bias)
        return LinearTransform