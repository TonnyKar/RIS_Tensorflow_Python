import tensorflow as tf
import math
import numpy as np
from sklearn.utils.linear_assignment_ import _hungarian

sess = tf.Session()


class MatchCriterion:

    def __init__(self, lambda_value):
        self.lambda_value = lambda_value
        self.gradInput = []
        self.gradInput_tensor = tf.convert_to_tensor(self.gradInput)
        self.criterion = tf.nn.sigmoid_cross_entropy_with_logits(labels=[], logits=[])
        self.gt_class = tf.placeholder(tf.float32, shape=[None])
        self.assignments = []
        self.who_min_ss = tf.placeholder(tf.float32, shape=[None])

    def IoU(self, x, y):
        a = tf.convert_to_tensor(x, dtype=tf.float32)
        x = a.eval()
        b = tf.convert_to_tensor(y, dtype=tf.float32)
        iou_inter = tf.reduce_sum(tf.multiply(a, b))
        iou_inter = iou_inter / (tf.reduce_sum(a) + tf.reduce_sum(b) - iou_inter)
        return iou_inter.eval()

    def updateOutput(self, dictionary, scores, ys):
        qs = dictionary
        ss = dictionary

        original_dimensionality = qs.shape[0]
        elements_prediction = qs.shape[1]
        elements_gt = 0
        ys_size = tf.size(ys)
        if ys_size is not 0:
            elements_gt = ys.shape[1]
        zero = tf.zeros([1])
        one = tf.ones([1])
        M_size = elements_gt
        M_tensor = tf.zeros([M_size, M_size])
        M = M_tensor.eval()
        ys = ys.eval()
        qs = dictionary.eval()
        ss = scores.eval()

        for i in range(0, min(elements_prediction, elements_gt)):
            for j in range(0, elements_gt):
                M[i, j] = -self.IoU(qs[:, i], ys[:, j])
                temp = self.lambda_value * (tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.convert_to_tensor(ss[i, :]), labels=one))
                M[i, j] = M[i, j] + temp.eval()

        if elements_gt > 0:
            self.assignments = _hungarian(M)
        output = 0
        # Save May Perform #
        for row, column in self.assignments:
            output = output + M[row, column]
        for i in range(elements_gt + 1, elements_prediction):
            temp = self.lambda_value * tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.convert_to_tensor(ss[i, :]), labels=zero)
            output = output + temp.eval()
        return output

    def updateGradInput(self, qs_and_ss_tensor, ys_tensor):
        qs_and_ss = qs_and_ss_tensor.eval()
        ys = ys_tensor.eval()
        qs = qs_and_ss[0]  # Tensor #
        ss = qs_and_ss[1]  # Tensor #
        grad_qs = np.full(np.shape(qs), 0)
        grad_ss = np.full(np.shape(ss), 0)
        original_dimensionality = qs.shape[0]
        elements_prediction = qs.shape[1]
        elements_gt = 0
        if (ys.size > 0):
            elements_gt = ys.shape[1]

        zero = tf.zeros([1])
        one = tf.ones([1])
        assignment_column = []
        for row, column in self.assignments:
            assignment_column[row] = column
        for i in range(0, elements_prediction):
            if i <= elements_gt:
                q = qs[:, i]
                y = ys[:, assignment_column]
                num = np.transpose(q).dot(y)
                num = np.resize(num, np.size(num))
                den = -num
                aux2 = np.transpose(q).dot(y)
                aux2 = aux2[1][1]
                q_tensor = tf.convert_to_tensor(q)
                y_tensor = tf.convert_to_tensor(y)
                aux2_tensor = tf.convert_to_tensor(aux2)
                aux_tensor = tf.reduce_sum(q_tensor) + tf.reduce_sum(y_tensor) - aux2_tensor
                aux = aux_tensor.eval()
                den = aux
                num = num[1]
                aux_den = np.full(np.shape(y), den)
                aux_ones = np.ones(np.shape(y))
                aux_num = np.full(np.shape(y), num)

                aux_den_tensor = tf.convert_to_tensor(aux_den)
                aux_ones_tensor = tf.convert_to_tensor(aux_ones)
                aux_num_tensor = tf.convert_to_tensor(aux_num)

                aux1_tensor = tf.multiply(aux_den_tensor, y_tensor)
                aux_tensor = -(aux1_tensor - tf.multiply(aux_ones_tensor - y_tensor, aux_num_tensor))

                aux = aux_tensor.eval()
                aux_den2 = np.full(np.shape(aux), den ** 2)
                aux_den2_tensor = tf.convert_to_tensor(aux_den2)

                aux_tensor = tf.divide(aux_tensor, aux_den2_tensor)
                aux = aux_tensor.eval()

                grad_qs[:, i] = np.resize(aux, np.size(aux))

                grad_ss_tensor = self.lambda_value * tf.gradients(tf.nn.sigmoid_cross_entropy_with_logits
                                                                  (logits=tf.convert_to_tensor(ss[i, :], labels=one)))

                grad_ss[i] = grad_ss_tensor.eval()

            else:
                grad_ss_tensor = self.lambda_value * tf.gradients(tf.nn.sigmoid_cross_entropy_with_logits
                                                                  (logits=tf.convert_to_tensor(ss[i, :], labels=zero)))

                grad_ss[i] = grad_ss_tensor.eval()

        self.gradInput.append(grad_qs)
        self.gradInput.append(grad_ss)
        self.gradInput_tensor = tf.convert_to_tensor(self.gradInput) # Will generate error

        return self.gradInput_tensor
