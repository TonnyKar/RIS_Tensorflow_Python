import tensorflow as tf
import numpy as np
from create_cnn import Create_CNN
from ConvLSTM import ConvLSTM
from plants_util import PlantUtils
from post_lstm import Post_LSTM
from MatchCriterion import MatchCriterion
import os


data_directory_home = "Path_To_Data/Data_Folder_Name"

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
iterations = 10000
summary_after = 128
input_path = data_directory_home
gt_path = data_directory_home

xSize = 106
ySize = 100
ht_array = np.zeros(shape=[1, 100, 106, 30])
ct_array = np.zeros(shape=[1, 100, 106, 30])
counter = 0


def Training(input_image, label_image):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            sess.as_default()
            sess.run(tf.global_variables_initializer())
            global ht_array
            global ct_array
            global counter
            Dictionary = []
            Scores = []

            i = PlantUtils().create_instance(input_image, label_image, 500, 530, 100, 106, 1, 1, 1)
            input_image = i[0]
            gt_image = i[1]
            c_mask = i[2]

            '''  Fully Connected Network and Convolution Network    '''

            fcn = Create_CNN().create_FCNN(input=input_image, kernelSize=9, kernelStride=5, nChannels=30)
            rnn_cnn = Create_CNN().create_cnn(input=fcn, kernelSize=3, kernelStride=1, nChannels=30)

            '''  ConvLSTM implementation and output  '''

            xt = rnn_cnn
            ht = tf.zeros(rnn_cnn.shape)
            ct = tf.zeros(rnn_cnn.shape)

            for j in range(0, min(seq_length, gt_image.shape[2])):
                for i in range(0, rnn_layers):
                    ct_ht = ConvLSTM().convlstm(xt, ht, ct)
                    ct = ct_ht[0] # Updated Ct
                    ht = ct_ht[1] # Update ht

                '''  Dictionary and Scores update  '''

                Dictionary.append(Post_LSTM().dictionary(tranined_input=ht, upsamplingScale=5).eval())
                test = (Post_LSTM().scores(trained_input=ht, ksize=2).eval())
                Scores.append(np.ravel(test))

            #ht_array = ht.eval()
            #ct_array = ct.eval()

            #counter += 1
            #print Scores

            Dictionary = np.asarray(Dictionary)
            Scores = np.asarray(Scores)
            #print Scores

            #print np.shape(Dictionary)


            Dictionary = tf.transpose(tf.convert_to_tensor(Dictionary))
            Scores = tf.convert_to_tensor(Scores)

            #print Dictionary

            '''  Loss Calculation  '''

            target_width = gt_image.shape[1]
            target_height = gt_image.shape[2]
            resize_target_shape = target_width * target_height
            target = gt_image
            target = tf.transpose(tf.reshape(target, shape=[gt_image.shape[0], resize_target_shape]))

            loss = MatchCriterion(lambda_value).updateOutput(Dictionary, Scores, target)
            print('loss', loss)
            loss = tf.Variable(tf.convert_to_tensor(loss))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss)
            sess.run(tf.global_variables_initializer())
            f = sess.run(train_op)

            correct_pred = tf.equal(tf.argmax(Dictionary, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            print('accuracy', accuracy.eval())

            '''
            sess.run(Dictionary)
            saver = tf.train.Saver()
            save_path = saver.save(sess, "Path_To_Model/model.ckpt")
            print("Dictionary saved in path: %s" % save_path)
            '''


'''
data_path = os.path.join(data_directory_home, 'data_list.txt')
label_path = os.path.join(data_directory_home, 'label_list.txt')
input_file = open(data_path, 'r')
input_IMAGE = input_file.readlines()

label_file = open(label_path, 'r')
label_IMAGE = label_file.readlines()

'''
for i in range(0, iterations):
    print('Step', i)
    Training(input_image='/Path_To_Data/Image1.png',
             label_image='/Path_To_Data/Label1.png')
    