import tensorflow as tf
from tensorflow.contrib.image import rotate
import numpy


class PlantUtils:

    def create_instance(self, input_image, gt_image, width, height, gt_width, gt_height, scaling, flipping, rotating ):
        original_height = 530
        original_width = 500

        file_learn = tf.train.string_input_producer([input_image])
        gt_file = tf.train.string_input_producer([gt_image])

        reader = tf.WholeFileReader()
        learn_key, learn_value = reader.read(file_learn)
        gt_key, gt_value = reader.read(gt_file)

        image_learn = tf.image.decode_png(learn_value)
        image_gt = tf.image.decode_png(gt_value)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_learn.eval()
        image_gt.eval()

        new_height_original = tf.to_int32(original_height)
        new_width_original = tf.to_int32(original_width)

        image_learn = tf.image.resize_images(image_learn, [new_width_original, new_height_original])
        image_gt = tf.image.resize_images(image_gt, [new_width_original, new_height_original])

        image_learn = tf.convert_to_tensor(image_learn.eval())
        image_gt = tf.convert_to_tensor(image_gt.eval())

        image_learn = tf.image.flip_left_right(image_learn)
        image_gt = tf.image.flip_left_right(image_gt)

        image_learn = rotate(image_learn, angles=360, interpolation='BILINEAR')
        image_gt = rotate(image_gt, angles=360, interpolation='NEAREST')

        new_height_learn = tf.to_int32(height)
        new_width_learn = tf.to_int32(width)

        new_height_gt = tf.to_int32(gt_height)
        new_width_gt = tf.to_int32(gt_width)

        image_learn = tf.image.resize_images(image_learn, [new_width_learn, new_height_learn])
        image_gt = tf.image.resize_images(image_gt, [new_width_gt, new_height_gt])

        n_instances = tf.reduce_max(image_gt)
        n_instances = n_instances.eval()
        numberofInstancesPerImage = n_instances
        nPixels = width * height

        gt_tensor = tf.zeros([n_instances, gt_width, gt_height], tf.int32)
        gt_tensor_value = gt_tensor.eval()
        max_val = tf.to_float(numberofInstancesPerImage)

        for i in range(0, n_instances):
            current_mask_value = gt_tensor_value[i, :, :]
            current_mask_value[tf.cast(tf.greater_equal(image_gt, max_val), tf.int32).eval()] = 1
            gt_tensor_value[tf.cast(tf.greater_equal(image_gt, max_val), tf.int32).eval()] = 0
        current_mask = tf.convert_to_tensor(current_mask_value)
        image_gt = tf.convert_to_tensor(gt_tensor_value)

        coord.request_stop()
        coord.join(threads)

        return image_learn, image_gt, current_mask





