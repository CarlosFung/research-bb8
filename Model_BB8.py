import tensorflow as tf
import numpy as np
from layers import *
import cv2


class Model_BB8:

    # the number of output classes, note that one class must be the background
    number_of_classes = 16

    # number of outputs for BB corners
    number_BB_outputs = 16

    # prediction output of the first network
    X_pred = []

    # Model endpoints
    segmentation_logits = [] # segmentation activations
    seg_predictions = [] # segmentation results from the first network (argmax)
    BB8_coordinates = []  # bounding box 8 corners prediction


    # Constructor to initialize the object
    def __init__(self, number_of_classes = 16, number_BB_outputs = 16):
        self.number_of_classes = number_of_classes
        self.number_BB_outputs = number_BB_outputs

    # Forward pass/Inference
    # :param input_op: (array) Placeholder for RGB image input as array of size [N, width, height, 3]. Pixel range is [0, 255]
    # :param seg_pred: (array) Placeholder to funnel the predictions from Stage 1 into Stage 2 of size [N, width, height, 1]
    # :param keep_prob_seg: (float) Dropout, probability to keep the values. For stage 1 only.
    # :param keep_prob_pose_conv/keep_prob_pose_hidden: (float) Dropout, probability to keep the values, For stage 2 only.
    # :return: The three graph endpoints (tensorflow nodes)
    #         self.segmentation_logits - The activation outputs of stage 1 of size [N, width*height, C]
    #         self.seg_predictions - The prediction output of stage 1 of size [N, width, height, C], each pixel contains a class label.
    #         self.BB8_coordinates - The bounding box corners prediction graph
    def inference_op(self, input_op, seg_pred, keep_prob_seg, keep_prob_pose_conv, keep_prob_pose_hidden):
        # upscale factor for the Deconvolution
        upscale = 2

        width = input_op.shape[1]  # 128
        height = input_op.shape[2]  # 128

        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        # First stage: Segmentation network

        # Convolution
        # input: 128x128x3
        # block 1 -- outputs 64x64x64
        conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
        conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
        pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)
        conv1_drop = tf.nn.dropout(pool1, keep_prob_seg, name="conv1_drop")

        # block 2 -- outputs 32x32x128
        conv2_1 = conv_op(conv1_drop, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
        conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
        pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)
        conv2_drop = tf.nn.dropout(pool2, keep_prob_seg, name="conv2_drop")

        # block 3 -- outputs 16x16x256
        conv3_1 = conv_op(conv2_drop, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
        conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
        conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1)
        pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)
        conv3_drop = tf.nn.dropout(pool3, keep_prob_seg, name="conv3_drop")

        # block 4 -- outputs 8x8x512
        conv4_1 = conv_op(conv3_drop, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
        conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
        conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
        pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)
        conv4_drop = tf.nn.dropout(pool4, keep_prob_seg, name="conv4_drop")

        # Deconvolution
        # block 5 -- outputs 16x16x256
        deconv5 = upsample_op(conv4_drop, "deconv5", 512, upscale)
        pconv5 = conv_op(deconv5, name="pw5", kh=1, kw=1, n_out=256, dh=1, dw=1)

        # block 6 -- outputs 32x32x128
        deconv6 = upsample_op(pconv5, "deconv6", 256, upscale)
        pconv6 = conv_op(deconv6, name="pw6", kh=1, kw=1, n_out=128, dh=1, dw=1)

        # block 7 -- outputs 64x64x64
        deconv7 = upsample_op(pconv6, "deconv7", 128, upscale)
        pconv7 = conv_op(deconv7, name="pw7", kh=1, kw=1, n_out=64, dh=1, dw=1)

        # block 8 -- outputs 128x128x16
        deconv8 = upsample_op(pconv7, "deconv8", 64, upscale)
        seg_result = conv_op(deconv8, name="pw8", kh=1, kw=1, n_out=self.number_of_classes, dh=1, dw=1)

        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        # For image size (128x128), the last layer is of size [N,128,128,16] -> [N, 16384, 16]
        # with number_of_classes = 16
        self.segmentation_logits = tf.reshape(tensor=seg_result, shape=(-1, width * height, self.number_of_classes))

        # Segmentation Network Output: (128x128)x16

        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        # First to second stage transition:
        # 1. Find the object
        # 2. Calculate its 2D center
        # 3. Calculate its 2D ROI
        # 4. Resize ROI to 16x16

        # seg_result: [N,128,128,16]
        # -> result_max: [N,128,128,1]
        result_max = tf.argmax(seg_result, 3)  # max result along axis 3

        # Non-background
        # size: [N,128,128,1]
        result_max_labels = tf.not_equal(result_max, 0)

        # result_max_result: [N,128,128,1]
        result_max_result = tf.cast(result_max_labels, tf.float32)
        # self.seg_predictions: [N,128,128,1]
        self.seg_predictions = tf.reshape(tensor=result_max_result, shape=(-1, width, height, 1))


        # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------
        # Second stage: BB8 estimation network

        # segment the results from the network
        pose_input_op = tf.multiply(seg_pred, input_op)

        # input: 16x16x3
        # block 1 -- outputs 8x8x64
        conv11_1 = conv_op(pose_input_op, name="conv11_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
        conv11_2 = conv_op(conv11_1, name="conv11_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
        pool11 = mpool_op(conv11_2, name="pool11", kh=2, kw=2, dw=2, dh=2)
        conv11_drop = tf.nn.dropout(pool11, keep_prob_pose_conv, name="conv11_drop")

        # block 2 -- outputs 4x4x128
        conv12_1 = conv_op(conv11_drop, name="conv12_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
        conv12_2 = conv_op(conv12_1, name="conv12_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
        pool12 = mpool_op(conv12_2, name="pool12", kh=2, kw=2, dh=2, dw=2)
        conv12_drop = tf.nn.dropout(pool12, keep_prob_pose_conv, name="conv12_drop")

        # block 3 -- outputs 2x2x256
        conv13_1 = conv_op(conv12_drop, name="conv13_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
        conv13_2 = conv_op(conv13_1, name="conv13_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
        conv13_3 = conv_op(conv13_2, name="conv13_3", kh=3, kw=3, n_out=256, dh=1, dw=1)
        pool13 = mpool_op(conv13_3, name="pool13", kh=2, kw=2, dh=2, dw=2)
        conv13_drop = tf.nn.dropout(pool13, keep_prob_pose_conv, name="conv13_drop")

        # block 4 -- outputs 2x2x256
        conv14_1 = conv_op(conv13_drop, name="conv14_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
        conv14_2 = conv_op(conv14_1, name="conv14_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
        conv14_3 = conv_op(conv14_2, name="conv14_3", kh=3, kw=3, n_out=256, dh=1, dw=1)
        pool14 = mpool_op(conv14_3, name="pool14", kh=2, kw=2, dh=2, dw=2)
        conv14_drop = tf.nn.dropout(pool14, keep_prob_pose_conv, name="conv14_drop")

        # flatten
        # 2x2x256 = 1024
        # reshape -- outputs 1x1024
        shp = conv14_drop.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        reshp = tf.reshape(conv14_drop, [-1, flattened_shape], name="reshp")

        # fully connected
        # fc 1 -- outputs 1x4096
        fc15 = fc_op(reshp, name="fc15", n_out=4096)
        fc15_drop = tf.nn.dropout(fc15, keep_prob_pose_hidden, name="fc15_drop")

        # fc 2 -- outputs 1x4096
        fc16 = fc_op(fc15_drop, name="fc16", n_out=4096)
        fc16_drop = tf.nn.dropout(fc16, keep_prob_pose_hidden, name="fc16_drop")

        # fc 3 -- outputs 1x16
        BB8_coordinates_predicted = fc_op(fc16_drop, name="fc17", n_out=self.number_BB_outputs)
        self.BB8_coordinates = tf.reshape(tensor=BB8_coordinates_predicted, shape=(-1, self.number_BB_outputs))
        # Second Stage Output: 1x16 (x1,y1,x2,y2,...,x16,y16) for each object

        #            [N,128x128,16]           [N,128,128,1]     [N,16], i.e.(x1,y1,x2,y2,...,x8,y8)
        return self.segmentation_logits, self.seg_predictions, self.BB8_coordinates