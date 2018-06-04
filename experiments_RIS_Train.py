import tensorflow as tf
import numpy as np
from create_cnn import Create_CNN
from ConvLSTM import ConvLSTM
from plants_util import PlantUtils
from post_lstm import Post_LSTM
from MatchCriterion import MatchCriterion
import os
import gc


data_directory_home = "/Path_To_Data/Data_Folder_Name"

original_height = 530
original_width = 500

name = "RIS_Experiment"
pre_model = "Pre-CNN model"
cnn_model = "CNN Model"
lstm_model = "LSTM model"
seq_length = 2
lambda_value = 1
pass_out = 0
non_object_iterations = 1
image_height = 530
image_width = 500
learn_pre = 0
learn_cnn = 1
learn_lstm = 1
learn_post_lstm = 1
learning_rate = 10**-4
rnn_channels = 30
rnn_layers = 2
rnn_filter_size = 3
cnn_filter_size = 3
iterations = 100000
summary_after = 128
input_path = data_directory_home
gt_path = data_directory_home

xSize = 106
ySize = 100
ht_array = np.zeros(shape=[1, 100, 106, 30])
ct_array = np.zeros(shape=[1, 100, 106, 30])
counter = 0
dictionary = []
scores = []
D = tf.placeholder(dtype=tf.float32, shape=(10600, 2))
S = tf.placeholder(dtype=tf.float32, shape=(2, 1))
T = tf.placeholder(dtype=tf.float32, shape=(10600, None))

graph1 = tf.Graph()

with graph1.as_default():
    X = tf.placeholder(dtype=tf.float32, shape=(500, 530, 3))
    Y = tf.placeholder(dtype=tf.float32, shape=(None, 100, 106))

    '''  Fully Connected Network and Convolution Network    '''

    fcn = Create_CNN().create_FCNN(input=X, kernelSize=9, kernelStride=5, nChannels=30)
    rnn_cnn = Create_CNN().create_cnn(input=fcn, kernelSize=3, kernelStride=1, nChannels=30)

    '''  ConvLSTM implementation and output  '''

    xt = rnn_cnn
    ht = tf.zeros(rnn_cnn.shape)
    ct = tf.zeros(rnn_cnn.shape)

    for j in range(0, min(seq_length, Y.shape[2])):
        for i in range(0, rnn_layers):
            ct_ht = ConvLSTM().convlstm(xt, ht, ct)
            ct = ct_ht[0]  # Updated Ct
            ht = ct_ht[1]  # Update ht
        dictionary.append(Post_LSTM().dictionary(tranined_input=ht, upsamplingScale=5))
        test = Post_LSTM().scores(trained_input=ht, ksize=2)
        scores.append(test)
    dictionary = tf.transpose(dictionary)

name1 = ['/Path_To_Data/Image001_rgb.png',
        '/Path_To_Data/Image002_rgb.png',
        '/Path_To_Data/Image003_rgb.png',
        '/Path_To_Data/Image004_rgb.png',
        '/Path_To_Data/Image005_rgb.png',
        '/Path_To_Data/Image006_rgb.png',
        '/Path_To_Data/Image007_rgb.png',
        '/Path_To_Data/Image008_rgb.png',
        '/Path_To_Data/Image009_rgb.png',
        '/Path_To_Data/Image010_rgb.png',
        '/Path_To_Data/Image011_rgb.png',
        '/Path_To_Data/Image012_rgb.png',
        '/Path_To_Data/Image013_rgb.png',
        '/Path_To_Data/Image014_rgb.png',
        '/Path_To_Data/Image015_rgb.png',
        '/Path_To_Data/Image016_rgb.png',
        '/Path_To_Data/Image017_rgb.png',
        '/Path_To_Data/Image018_rgb.png',
        '/Path_To_Data/Image019_rgb.png',
        '/Path_To_Data/Image020_rgb.png',
        '/Path_To_Data/Image021_rgb.png',
        '/Path_To_Data/Image022_rgb.png',
        '/Path_To_Data/Image023_rgb.png',
        '/Path_To_Data/Image024_rgb.png',
        '/Path_To_Data/Image025_rgb.png',
        '/Path_To_Data/Image026_rgb.png',
        '/Path_To_Data/Image027_rgb.png',
        '/Path_To_Data/Image028_rgb.png',
        '/Path_To_Data/Image029_rgb.png',
        '/Path_To_Data/Image030_rgb.png',
        '/Path_To_Data/Image031_rgb.png',
        '/Path_To_Data/Image032_rgb.png',
        '/Path_To_Data/Image033_rgb.png',
        '/Path_To_Data/Image034_rgb.png',
        '/Path_To_Data/Image035_rgb.png',
        '/Path_To_Data/Image036_rgb.png',
        '/Path_To_Data/Image037_rgb.png',
        '/Path_To_Data/Image038_rgb.png',
        '/Path_To_Data/Image039_rgb.png',
        '/Path_To_Data/Image040_rgb.png',
        '/Path_To_Data/Image041_rgb.png',
        '/Path_To_Data/Image042_rgb.png',
        '/Path_To_Data/Image043_rgb.png',
        '/Path_To_Data/Image044_rgb.png',
        '/Path_To_Data/Image045_rgb.png',
        '/Path_To_Data/Image046_rgb.png',
        '/Path_To_Data/Image047_rgb.png',
        '/Path_To_Data/Image048_rgb.png',
        '/Path_To_Data/Image049_rgb.png',
        '/Path_To_Data/Image050_rgb.png',
        '/Path_To_Data/Image051_rgb.png',
        '/Path_To_Data/Image052_rgb.png',
        '/Path_To_Data/Image053_rgb.png',
        '/Path_To_Data/Image054_rgb.png',
        '/Path_To_Data/Image055_rgb.png',
        '/Path_To_Data/Image056_rgb.png',
        '/Path_To_Data/Image057_rgb.png',
        '/Path_To_Data/Image058_rgb.png',
        '/Path_To_Data/Image059_rgb.png',
        '/Path_To_Data/Image060_rgb.png',
        '/Path_To_Data/Image061_rgb.png',
        '/Path_To_Data/Image062_rgb.png',
        '/Path_To_Data/Image063_rgb.png',
        '/Path_To_Data/Image064_rgb.png',
        '/Path_To_Data/Image065_rgb.png',
        '/Path_To_Data/Image066_rgb.png',
        '/Path_To_Data/Image067_rgb.png',
        '/Path_To_Data/Image068_rgb.png',
        '/Path_To_Data/Image069_rgb.png',
        '/Path_To_Data/Image070_rgb.png',
        '/Path_To_Data/Image071_rgb.png',
        '/Path_To_Data/Image072_rgb.png',
        '/Path_To_Data/Image073_rgb.png',
        '/Path_To_Data/Image074_rgb.png',
        '/Path_To_Data/Image075_rgb.png',
        '/Path_To_Data/Image076_rgb.png',
        '/Path_To_Data/Image077_rgb.png',
        '/Path_To_Data/Image078_rgb.png',
        '/Path_To_Data/Image079_rgb.png',
        '/Path_To_Data/Image080_rgb.png',
        '/Path_To_Data/Image081_rgb.png',
        '/Path_To_Data/Image082_rgb.png',
        '/Path_To_Data/Image083_rgb.png',
        '/Path_To_Data/Image084_rgb.png',
        '/Path_To_Data/Image085_rgb.png',
        '/Path_To_Data/Image086_rgb.png',
        '/Path_To_Data/Image087_rgb.png',
        '/Path_To_Data/Image088_rgb.png',
        '/Path_To_Data/Image089_rgb.png',
        '/Path_To_Data/Image090_rgb.png',
        '/Path_To_Data/Image091_rgb.png',
        '/Path_To_Data/Image092_rgb.png',
        '/Path_To_Data/Image093_rgb.png',
        '/Path_To_Data/Image094_rgb.png',
        '/Path_To_Data/Image095_rgb.png',
        '/Path_To_Data/Image096_rgb.png',
        '/Path_To_Data/Image097_rgb.png',
        '/Path_To_Data/Image098_rgb.png',
        '/Path_To_Data/Image099_rgb.png',
        '/Path_To_Data/Image100_rgb.png',
        '/Path_To_Data/Image101_rgb.png',
        '/Path_To_Data/Image102_rgb.png',
        '/Path_To_Data/Image103_rgb.png',
        '/Path_To_Data/Image104_rgb.png',
        '/Path_To_Data/Image105_rgb.png',
        '/Path_To_Data/Image106_rgb.png',
        '/Path_To_Data/Image107_rgb.png',
        '/Path_To_Data/Image108_rgb.png',
        '/Path_To_Data/Image109_rgb.png',
        '/Path_To_Data/Image110_rgb.png',
        '/Path_To_Data/Image111_rgb.png',
        '/Path_To_Data/Image112_rgb.png',
        '/Path_To_Data/Image113_rgb.png',
        '/Path_To_Data/Image114_rgb.png',
        '/Path_To_Data/Image115_rgb.png',
        '/Path_To_Data/Image116_rgb.png',
        '/Path_To_Data/Image117_rgb.png',
        '/Path_To_Data/Image118_rgb.png',
        '/Path_To_Data/Image119_rgb.png',
        '/Path_To_Data/Image120_rgb.png',
        '/Path_To_Data/Image121_rgb.png',
        '/Path_To_Data/Image122_rgb.png',
        '/Path_To_Data/Image123_rgb.png',
        '/Path_To_Data/Image124_rgb.png',
        '/Path_To_Data/Image125_rgb.png',
        '/Path_To_Data/Image126_rgb.png',
        '/Path_To_Data/Image127_rgb.png',
        '/Path_To_Data/Image128_rgb.png']

name2 = ['/Path_To_Data/Image001_label.png',
        '/Path_To_Data/Image002_label.png',
        '/Path_To_Data/Image003_label.png',
        '/Path_To_Data/Image004_label.png',
        '/Path_To_Data/Image005_label.png',
        '/Path_To_Data/Image006_label.png',
        '/Path_To_Data/Image007_label.png',
        '/Path_To_Data/Image008_label.png',
        '/Path_To_Data/Image009_label.png',
        '/Path_To_Data/Image010_label.png',
        '/Path_To_Data/Image011_label.png',
        '/Path_To_Data/Image012_label.png',
        '/Path_To_Data/Image013_label.png',
        '/Path_To_Data/Image014_label.png',
        '/Path_To_Data/Image015_label.png',
        '/Path_To_Data/Image016_label.png',
        '/Path_To_Data/Image017_label.png',
        '/Path_To_Data/Image018_label.png',
        '/Path_To_Data/Image019_label.png',
        '/Path_To_Data/Image020_label.png',
        '/Path_To_Data/Image021_label.png',
        '/Path_To_Data/Image022_label.png',
        '/Path_To_Data/Image023_label.png',
        '/Path_To_Data/Image024_label.png',
        '/Path_To_Data/Image025_label.png',
        '/Path_To_Data/Image026_label.png',
        '/Path_To_Data/Image027_label.png',
        '/Path_To_Data/Image028_label.png',
        '/Path_To_Data/Image029_label.png',
        '/Path_To_Data/Image030_label.png',
        '/Path_To_Data/Image031_label.png',
        '/Path_To_Data/Image032_label.png',
        '/Path_To_Data/Image033_label.png',
        '/Path_To_Data/Image034_label.png',
        '/Path_To_Data/Image035_label.png',
        '/Path_To_Data/Image036_label.png',
        '/Path_To_Data/Image037_label.png',
        '/Path_To_Data/Image038_label.png',
        '/Path_To_Data/Image039_label.png',
        '/Path_To_Data/Image040_label.png',
        '/Path_To_Data/Image041_label.png',
        '/Path_To_Data/Image042_label.png',
        '/Path_To_Data/Image043_label.png',
        '/Path_To_Data/Image044_label.png',
        '/Path_To_Data/Image045_label.png',
        '/Path_To_Data/Image046_label.png',
        '/Path_To_Data/Image047_label.png',
        '/Path_To_Data/Image048_label.png',
        '/Path_To_Data/Image049_label.png',
        '/Path_To_Data/Image050_label.png',
        '/Path_To_Data/Image051_label.png',
        '/Path_To_Data/Image052_label.png',
        '/Path_To_Data/Image053_label.png',
        '/Path_To_Data/Image054_label.png',
        '/Path_To_Data/Image055_label.png',
        '/Path_To_Data/Image056_label.png',
        '/Path_To_Data/Image057_label.png',
        '/Path_To_Data/Image058_label.png',
        '/Path_To_Data/Image059_label.png',
        '/Path_To_Data/Image060_label.png',
        '/Path_To_Data/Image061_label.png',
        '/Path_To_Data/Image062_label.png',
        '/Path_To_Data/Image063_label.png',
        '/Path_To_Data/Image064_label.png',
        '/Path_To_Data/Image065_label.png',
        '/Path_To_Data/Image066_label.png',
        '/Path_To_Data/Image067_label.png',
        '/Path_To_Data/Image068_label.png',
        '/Path_To_Data/Image069_label.png',
        '/Path_To_Data/Image070_label.png',
        '/Path_To_Data/Image071_label.png',
        '/Path_To_Data/Image072_label.png',
        '/Path_To_Data/Image073_label.png',
        '/Path_To_Data/Image074_label.png',
        '/Path_To_Data/Image075_label.png',
        '/Path_To_Data/Image076_label.png',
        '/Path_To_Data/Image077_label.png',
        '/Path_To_Data/Image078_label.png',
        '/Path_To_Data/Image079_label.png',
        '/Path_To_Data/Image080_label.png',
        '/Path_To_Data/Image081_label.png',
        '/Path_To_Data/Image082_label.png',
        '/Path_To_Data/Image083_label.png',
        '/Path_To_Data/Image084_label.png',
        '/Path_To_Data/Image085_label.png',
        '/Path_To_Data/Image086_label.png',
        '/Path_To_Data/Image087_label.png',
        '/Path_To_Data/Image088_label.png',
        '/Path_To_Data/Image089_label.png',
        '/Path_To_Data/Image090_label.png',
        '/Path_To_Data/Image091_label.png',
        '/Path_To_Data/Image092_label.png',
        '/Path_To_Data/Image093_label.png',
        '/Path_To_Data/Image094_label.png',
        '/Path_To_Data/Image095_label.png',
        '/Path_To_Data/Image096_label.png',
        '/Path_To_Data/Image097_label.png',
        '/Path_To_Data/Image098_label.png',
        '/Path_To_Data/Image099_label.png',
        '/Path_To_Data/Image100_label.png',
        '/Path_To_Data/Image101_label.png',
        '/Path_To_Data/Image102_label.png',
        '/Path_To_Data/Image103_label.png',
        '/Path_To_Data/Image104_label.png',
        '/Path_To_Data/Image105_label.png',
        '/Path_To_Data/Image106_label.png',
        '/Path_To_Data/Image107_label.png',
        '/Path_To_Data/Image108_label.png',
        '/Path_To_Data/Image109_label.png',
        '/Path_To_Data/Image110_label.png',
        '/Path_To_Data/Image111_label.png',
        '/Path_To_Data/Image112_label.png',
        '/Path_To_Data/Image113_label.png',
        '/Path_To_Data/Image114_label.png',
        '/Path_To_Data/Image115_label.png',
        '/Path_To_Data/Image116_label.png',
        '/Path_To_Data/Image117_label.png',
        '/Path_To_Data/Image118_label.png',
        '/Path_To_Data/Image119_label.png',
        '/Path_To_Data/Image120_label.png',
        '/Path_To_Data/Image121_label.png',
        '/Path_To_Data/Image122_label.png',
        '/Path_To_Data/Image123_label.png',
        '/Path_To_Data/Image124_label.png',
        '/Path_To_Data/Image125_label.png',
        '/Path_To_Data/Image126_label.png',
        '/Path_To_Data/Image127_label.png',
        '/Path_To_Data/Image128_label.png']

for step in range(0, iterations):
    with tf.Session(graph=graph1) as sess:
        sess.as_default()
        sess.run(tf.global_variables_initializer())
        print('step: ', step)
        i = PlantUtils().create_instance(name1[0], name2[0], 500, 530, 100, 106, 1, 1, 1)
        input_image = i[0].eval()
        gt_image = i[1].eval()

        d, s = sess.run([dictionary, scores], feed_dict={X: input_image, Y: gt_image})
        s = np.asarray(s)
        s.shape = (2, 1)

        target_width = gt_image.shape[1]
        target_height = gt_image.shape[2]
        resize_target_shape = target_width * target_height
        target = gt_image
        target = tf.transpose(tf.reshape(target, shape=[gt_image.shape[0], resize_target_shape]))
        dicti = tf.convert_to_tensor(d)
        sco = tf.convert_to_tensor(s)
        loss = MatchCriterion(lambda_value).updateOutput(dicti, sco, target)
        print ('loss', loss)
        loss = tf.Variable(tf.convert_to_tensor(loss))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)
        sess.run(tf.global_variables_initializer())
        f = sess.run(train_op)

        correct_pred = tf.equal(tf.argmax(dicti, 1), tf.argmax(target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        print('accuracy', accuracy.eval())

        sess.close()
        gc.collect()













    












