import tensorflow as tf
import numpy as np
import pickle
import os
from os import path
from Model_BB8 import *
from layers import *
import cv2


class TrainRGB:
    # training mini-batch size
    batch_size = 128
    # test size
    test_size = 256
    # num epochs to train
    num_epochs = 2000

    # global step variables
    epoch = []
    step = 0

    # models
    model = []  # a model instance -> model = model_cls()
    model_cls = [] # reference to the model class

    # Learning rate
    learning_rate = 0.0001

    # number of classes
    number_classes = 16

    # number of outputs
    num_BB8_outputs = 16

    imgWidth = 128
    imgHeight = 128

    # placeholders
    ph_rgb = []  # input RGB data of size [N, width, height, 3]
    ph_seg_pred = []  # prediction output of the first network of size [N, width, height, 1]
    ph_mask = []  # ground truth mask of size [N, width*height, C]
    ph_gt_bb8 = []  # ground truth bounding box 8 corners of size [N, 16]

    # Dropout ratio
    keep_prob_seg = []
    keep_prob_pose_conv = []
    keep_prob_pose_hidden = []

    # References to data
    # All those variables point to the data location
    train_rgb = []     # Training RGB data
    train_gt_mask = [] # Training ground truth mask
    train_gt_one_mask = []  # Training ground truth mask, one class only
    train_gt_bb8 = []  # Training ground truth bounding box 8 corners
    train_gt_pose = [] # Training ground truth pose

    test_rgb = []      # Testing RGB data
    test_gt_mask = []  # Testing ground truth mask
    test_gt_one_mask = []  # Testing ground truth mask, one class only
    test_gt_bb8 = []   # Testing ground truth bounding box 8 corners
    test_gt_pose = []  # Testing ground truth pose

    # predictions output
    pre_logits = [] # output of stage 1, activations
    seg_predictions = [] # output of stage 1, class predictions.
    pre_bb8 = []  # pose output of stage 2

    #---------------------------------------------------------
    # 1-stage solver
    cross_entropies = []
    cross_entropy_sum = []
    optimizer = []
    train_seg = []
    prediction = []
    probabilities = []

    # 2-stage solver
    loss_bb8 = []
    optimizer2 = []
    train_bb8 = []

    # ---------------------------------------------------------
    # To save the variables.
    # Need to be created after all variables have been created.
    saver = []
    saver_log_file = ""
    saver_log_folder = ""

    # restore the model or train a new one
    restore_model = False
    restore_from_file = ""  # keep this empty to not restore the model

    # for debugging, if enabled, the class will show some opencv
    # debug windows during the training process.
    show_sample = True


    def __init__(self, model_class, num_classes, num_BB8_outputs, learning_rate=0.0001):
        self.model_cls = model_class
        self.learning_rate = learning_rate
        self.number_classes = num_classes
        self.num_BB8_outputs = num_BB8_outputs


    # Initialize the model of the tensorflow graph.
    def init_graph(self, img_width, img_height, restore_from_file = ""):
        self.restore_from_file = restore_from_file

        self.epoch = tf.Variable(0, name='epoche', trainable=False)

        self.imgWidth = img_width
        self.imgHeight = img_height

        # placeholders
        # the input image of size
        # [128, 128, 3]
        self.ph_rgb = tf.placeholder("float", [None, img_width, img_height, 3])

        # predictions from the first network, input for the second network.
        self.ph_seg_pred = tf.placeholder("float", [None, img_width, img_height, 1])

        # vector for the output data.
        # The network generates an output layer for each class.
        # 16384 = 128 * 128
        # [N, width*height, C]
        self.ph_mask = tf.placeholder("float", [None, img_width * img_height, self.number_classes], name="gt_mask")

        # the bb8 ground truth
        self.ph_gt_bb8 = tf.placeholder("float", [None, self.num_BB8_outputs], name="gt_bb8")

        # Dropout ratio placeholder
        self.keep_prob_seg = tf.placeholder("float", name="keep_prob_seg")
        self.keep_prob_pose_conv = tf.placeholder("float", name="keep_prob_pose_conv")
        self.keep_prob_pose_hidden = tf.placeholder("float", name="keep_prob_pose_hidden")

        # This solver will create its own instance of the model
        self.model = self.model_cls(self.number_classes, self.num_BB8_outputs)

        # [N,128x128,16]       [N,128,128,1]   [N,16], i.e.(x1,y1,x2,y2,...,x8,y8)
        self.pre_logits, self.seg_predictions, self.pre_bb8 = \
            self.model.inference_op(self.ph_rgb, self.ph_seg_pred, self.keep_prob_seg,
                                    self.keep_prob_pose_conv, self.keep_prob_pose_hidden)

        # solver
        self.__initSolver__()

        # To save the variables.
        # Need to be created after all variables have been created.
        self.saver = tf.train.Saver()

        if len(restore_from_file) > 0:
            self.restore_model = True
            self.restore_from_file = restore_from_file

    # Start to train the model.
    def train(self, train_rgb, train_gt_mask, train_gt_one_mask, train_gt_bb8, train_gt_pose,
              test_rgb, test_gt_mask, test_gt_one_mask, test_gt_bb8, test_gt_pose):
        self.train_rgb = train_rgb   # Training RGB data [N, width, height, 3]
        self.train_gt_mask = train_gt_mask # Training ground truth mask [N, width, height, C]
        self.train_gt_one_mask = train_gt_one_mask # Training ground truth mask [N, width, height]
        self.train_gt_bb8 = train_gt_bb8  # Training ground truth bounding box 8 corners [N, 16]
        self.train_gt_pose = train_gt_pose  # Training ground truth pose [x, y, z, qx, qy, qz, qw]

        self.test_rgb = test_rgb   # Testing RGB data [N, width, height, 3]
        self.test_gt_mask = test_gt_mask # Testing ground truth mask [N, width, height, C]
        self.test_gt_one_mask = test_gt_one_mask # Testing ground truth mask [N, width, height]
        self.test_gt_bb8 = test_gt_bb8  # Testing ground truth bounding box 8 corners [N, 16]
        self.test_gt_pose = test_gt_pose  # Testing ground truth pose [x, y, z, qx, qy, qz, qw]

        # invokes training
        self.__start_train()

    # Start the evaluation of the current model.
    def eval(self, test_rgb, test_gt_mask, test_gt_pose):
        self.__start_eval(test_rgb, test_gt_mask, test_gt_pose)


    # Init the solver for the model.
    def __initSolver__(self):
        # -------------------------------------------------------------------------------------
        # Solver first stage

        # Softmax and cross-entropy to determine the loss
        self.cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=self.pre_logits,
                                                                       labels=self.ph_mask)
        # Reduce the sum of all errors. This will sum up all the
        # incorrect identified pixels as loss and reduce this number.
        self.cross_entropy_sum = tf.reduce_sum(self.cross_entropies)

        # Training with adam optimizer.
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # reduce the gradients.
        gradients = self.optimizer.compute_gradients(loss=self.cross_entropy_sum)

        # The training step.
        self.train_seg = self.optimizer.apply_gradients(grads_and_vars=gradients)

        # Prediction operations
        # Maximum arguments over all logits along dimension 2.
        # self.pre_logits: [N,128x128,16]
        # self.prediction: [N,128x128,1]
        self.prediction = tf.argmax(self.pre_logits, 2)

        # Probability operation for all logits
        self.probabilities = tf.nn.softmax(self.pre_logits)

        # -------------------------------------------------------------------------------
        # Solver second stage
        self.loss_bb8 = tf.reduce_mean(
            (tf.norm((self.pre_bb8[:, 0:2] - self.ph_gt_bb8[:, 0:2]), ord='euclidean', axis=1, keep_dims=True) +
             tf.norm((self.pre_bb8[:, 2:4] - self.ph_gt_bb8[:, 2:4]), ord='euclidean', axis=1, keep_dims=True) +
             tf.norm((self.pre_bb8[:, 4:6] - self.ph_gt_bb8[:, 4:6]), ord='euclidean', axis=1, keep_dims=True) +
             tf.norm((self.pre_bb8[:, 6:8] - self.ph_gt_bb8[:, 6:8]), ord='euclidean', axis=1, keep_dims=True) +
             tf.norm((self.pre_bb8[:, 8:10] - self.ph_gt_bb8[:, 8:10]), ord='euclidean', axis=1, keep_dims=True) +
             tf.norm((self.pre_bb8[:, 10:12] - self.ph_gt_bb8[:, 10:12]), ord='euclidean', axis=1, keep_dims=True) +
             tf.norm((self.pre_bb8[:, 12:14] - self.ph_gt_bb8[:, 12:14]), ord='euclidean', axis=1, keep_dims=True) +
             tf.norm((self.pre_bb8[:, 14:16] - self.ph_gt_bb8[:, 14:16]), ord='euclidean', axis=1, keep_dims=True))
            / 8)

        # Training with RMSProp optimizer.
        self.optimizer2 = tf.train.RMSPropOptimizer(self.learning_rate, 0.9)

        # reduce the gradients.
        gradients2 = self.optimizer2.compute_gradients(loss=self.loss_bb8)

        # The training step.
        self.train_bb8 = self.optimizer.apply_gradients(grads_and_vars=gradients2)

        # self.train_bb8 = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.loss_bb8)

        return self.train_seg, self.prediction, self.probabilities, self.train_bb8


    # Start the training procedure.
    def __start_train(self):
        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # start the session
        with tf.Session() as sess:
            sess.run(initop)

            if self.restore_model:
                saver = tf.train.import_meta_graph(self.saver_log_folder + self.restore_from_file)
                saver.restore(sess, tf.train.latest_checkpoint(self.saver_log_folder))
                print("Model restored at epoche ", sess.run('epoche:0'))

            print("Start training")
            start_idx = sess.run('epoche:0') + 1

            for i in range(self.num_epochs):
                # Count the steps manually
                self.step = i + start_idx
                assign_op = self.epoch.assign(i + start_idx)
                sess.run(assign_op)

                # ---------------------------------------------------------------------------------
                # Train
                training_batch = zip(range(0, len(self.train_rgb), self.batch_size),
                                     range(self.batch_size, len(self.train_rgb) + 1, self.batch_size))

                # shuffle the indices
                indices = list(range(0, len(self.train_rgb)))
                shuffled = np.random.permutation(indices)

                # run the batch
                train_cnt = 0
                train_precision_sum = 0
                train_recall_sum = 0
                train_accuracy_sum = 0
                train_loss_sum = 0
                train_bb8_loss_sum = 0
                for start, end in training_batch:
                    train_cnt = train_cnt + 1
                    train_indices = shuffled[start:end]
                    train_predict, train_prob, train_loss, train_bb8_loss = \
                        self.__train_step(sess, self.train_rgb[train_indices],
                                          self.train_gt_mask[train_indices],
                                          self.train_gt_bb8[train_indices],
                                          self.train_gt_one_mask[train_indices],
                                          1.0, 1.0, 1.0)
                                          # 0.8)

                    train_precision, train_recall, train_accuracy = self.__getAccuracy(self.train_gt_one_mask[train_indices],
                                                                                       train_predict,
                                                                                       "train_accuracy_" + str(i) + "_"
                                                                                       + str(train_cnt) + ".csv")
                    train_precision_sum = train_precision_sum + train_precision
                    train_recall_sum = train_recall_sum + train_recall
                    train_accuracy_sum = train_accuracy_sum + train_accuracy
                    train_loss_sum = train_loss_sum + train_loss
                    train_bb8_loss_sum = train_bb8_loss_sum + train_bb8_loss

                train_precision_avg = train_precision_sum / train_cnt
                train_recall_avg = train_recall_sum / train_cnt
                train_accuracy_avg = train_accuracy_sum / train_cnt
                train_loss_avg = train_loss_sum / train_cnt
                train_bb8_loss_avg = train_bb8_loss_sum / train_cnt

                # ---------------------------------------------------------------------------------
                # Test accuracy
                test_indices = np.arange(len(self.test_rgb))
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:self.test_size]
                test_predict, test_prob, test_loss, test_bb8_loss = \
                    self.__test_step(sess, self.test_rgb[test_indices], self.test_gt_mask[test_indices],
                                     self.test_gt_bb8[test_indices])

                test_precision, test_recall, test_accuracy = self.__getAccuracy(self.test_gt_one_mask[test_indices],
                                                                                test_predict,
                                                                                "test_accuracy_"+str(i)+".csv")
                print("Epoch,", self.step,
                      ",Train Loss,", train_loss_avg / (self.imgWidth*self.imgHeight),
                      ",Train Precison,", train_precision_avg,
                      ",Train Recall,", train_recall_avg,
                      ",Train Accuracy,", train_accuracy_avg,
                      ",Train BB8 Loss,", train_bb8_loss_avg,
                      ",Test Loss,", test_loss / (self.imgWidth*self.imgHeight),
                      ",Test Precison,", test_precision,
                      ",Test Recall,", test_recall,
                      ",Test Accuracy,", test_accuracy,
                      ",Test BB8 Loss,", test_bb8_loss)

                # Save and test all 10 iterations
                if i % 10 == 0:
                    self.saver.save(sess, self.saver_log_folder + self.saver_log_file, global_step=self.step)
                    print("Saved at step ", self.step)

            # save the last step
            self.saver.save(sess, self.saver_log_folder + self.saver_log_file, global_step=self.step)
            self.restore_from_file = self.saver_log_file + "-" + str(self.step) + ".meta"  # keeps the file so that the evaluation can restore the last model.


    # Start the network evaluation.
    def __start_eval(self, Xte_rgb, Yte_mask, Yte_pose):
        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # run the test session
        with tf.Session() as sess:
            sess.run(initop)




    # Execute one training step for the entire graph
    def __train_step(self, sess, rgb_batch, mask_batch, bb8_batch, mask_batch_gt,
                     keep_prob_seg, keep_prob_pose_conv, keep_prob_pose_hidden):
        # Train the first stage.
        sess.run(self.train_seg, feed_dict={self.ph_rgb: rgb_batch,
                                            self.ph_mask: mask_batch,
                                            self.keep_prob_seg: keep_prob_seg})
        train_prob, train_loss = sess.run([self.probabilities,
                                           self.cross_entropy_sum],
                                          feed_dict={self.ph_rgb: rgb_batch,
                                                     self.ph_mask: mask_batch,
                                                     self.keep_prob_seg: keep_prob_seg})

        # Generate the output from the first stage - predicted mask.
        output = sess.run(self.seg_predictions, feed_dict={self.ph_rgb: rgb_batch,
                                                           self.ph_mask: mask_batch,
                                                           self.keep_prob_seg: keep_prob_seg})

        # Train the second stage.
        sess.run(self.train_bb8, feed_dict={self.ph_seg_pred: output,
                                            self.ph_rgb: rgb_batch,
                                            self.ph_gt_bb8: bb8_batch,
                                            self.keep_prob_pose_conv: keep_prob_pose_conv,
                                            self.keep_prob_pose_hidden: keep_prob_pose_hidden})
        # Calculate bb8 loss.
        train_bb8_loss = sess.run(self.loss_bb8, feed_dict={self.ph_seg_pred: output,
                                                            self.ph_rgb: rgb_batch,
                                                            self.ph_gt_bb8: bb8_batch,
                                                            self.keep_prob_pose_conv: keep_prob_pose_conv,
                                                            self.keep_prob_pose_hidden: keep_prob_pose_hidden})

        # Generate the output from the second stage - predicted bb8.
        output_bb8 = sess.run(self.pre_bb8, feed_dict={self.ph_seg_pred: output,
                                                       self.ph_rgb: rgb_batch,
                                                       self.keep_prob_pose_conv: keep_prob_pose_conv,
                                                       self.keep_prob_pose_hidden: keep_prob_pose_hidden})

        # Shows an RGB test image and the segmentation output.
        if self.show_sample:
            test_img = output[0] * 255
            test_gt_mask = mask_batch_gt[0].copy()
            test_gt_mask = np.reshape(test_gt_mask, newshape=(32, 32)) * 255
            test_rgb_raw = rgb_batch[0].copy()
            test_rgb = rgb_batch[0].copy()
            test_rgb_bb8_p = rgb_batch[0].copy()
            test_rgb_bb8_gt = rgb_batch[0].copy()
            test_bb8 = np.array(output_bb8[0], dtype=np.int32)
            test_bb8_gt = np.array(bb8_batch[0], dtype=np.int32)

            p1 = (test_bb8[0], test_bb8[1])
            p2 = (test_bb8[2], test_bb8[3])
            p3 = (test_bb8[4], test_bb8[5])
            p4 = (test_bb8[6], test_bb8[7])
            p5 = (test_bb8[8], test_bb8[9])
            p6 = (test_bb8[10], test_bb8[11])
            p7 = (test_bb8[12], test_bb8[13])
            p8 = (test_bb8[14], test_bb8[15])
            p1gt = (test_bb8_gt[0], test_bb8_gt[1])
            p2gt = (test_bb8_gt[2], test_bb8_gt[3])
            p3gt = (test_bb8_gt[4], test_bb8_gt[5])
            p4gt = (test_bb8_gt[6], test_bb8_gt[7])
            p5gt = (test_bb8_gt[8], test_bb8_gt[9])
            p6gt = (test_bb8_gt[10], test_bb8_gt[11])
            p7gt = (test_bb8_gt[12], test_bb8_gt[13])
            p8gt = (test_bb8_gt[14], test_bb8_gt[15])

            blue = [255, 0, 0]
            green = [0, 255, 0]
            red = [0, 0, 255]
            yellow = [0, 255, 255]
            magenta = [255, 0, 255]
            cyan = [255, 255, 0]
            pink = [175, 175, 255]
            orange = [0, 127, 255]


            # cv2.circle(image, (x, y), radius, (B,G,R), thickness)
            cv2.circle(test_rgb, p1, 1, blue)
            cv2.circle(test_rgb, p2, 1, green)
            cv2.circle(test_rgb, p3, 1, red)
            cv2.circle(test_rgb, p4, 1, yellow)
            cv2.circle(test_rgb, p5, 1, magenta)
            cv2.circle(test_rgb, p6, 1, cyan)
            cv2.circle(test_rgb, p7, 1, pink)
            cv2.circle(test_rgb, p8, 1, orange)

            cv2.circle(test_rgb, p1gt, 1, blue, -1)
            cv2.circle(test_rgb, p2gt, 1, green, -1)
            cv2.circle(test_rgb, p3gt, 1, red, -1)
            cv2.circle(test_rgb, p4gt, 1, yellow, -1)
            cv2.circle(test_rgb, p5gt, 1, magenta, -1)
            cv2.circle(test_rgb, p6gt, 1, cyan, -1)
            cv2.circle(test_rgb, p7gt, 1, pink, -1)
            cv2.circle(test_rgb, p8gt, 1, orange, -1)

            cv2.circle(test_rgb_bb8_p, p1, 1, blue)
            cv2.circle(test_rgb_bb8_p, p2, 1, green)
            cv2.circle(test_rgb_bb8_p, p3, 1, red)
            cv2.circle(test_rgb_bb8_p, p4, 1, yellow)
            cv2.circle(test_rgb_bb8_p, p5, 1, magenta)
            cv2.circle(test_rgb_bb8_p, p6, 1, cyan)
            cv2.circle(test_rgb_bb8_p, p7, 1, pink)
            cv2.circle(test_rgb_bb8_p, p8, 1, orange)

            cv2.circle(test_rgb_bb8_gt, p1gt, 1, blue, -1)
            cv2.circle(test_rgb_bb8_gt, p2gt, 1, green, -1)
            cv2.circle(test_rgb_bb8_gt, p3gt, 1, red, -1)
            cv2.circle(test_rgb_bb8_gt, p4gt, 1, yellow, -1)
            cv2.circle(test_rgb_bb8_gt, p5gt, 1, magenta, -1)
            cv2.circle(test_rgb_bb8_gt, p6gt, 1, cyan, -1)
            cv2.circle(test_rgb_bb8_gt, p7gt, 1, pink, -1)
            cv2.circle(test_rgb_bb8_gt, p8gt, 1, orange, -1)

            cv2.imshow("test_img", test_img)
            cv2.imshow("test_gt_mask", test_gt_mask)
            cv2.imshow("test_rgb", test_rgb_raw)
            cv2.imshow("test_bb8", test_rgb)
            cv2.imshow("test_bb8_pred", test_rgb_bb8_p)
            cv2.imshow("test_bb8_gt", test_rgb_bb8_gt)

            cv2.moveWindow('test_img', 30, 450)
            cv2.moveWindow('test_gt_mask', 230, 450)
            cv2.moveWindow('test_rgb', 30, 560)
            cv2.moveWindow('test_bb8', 30, 670)
            cv2.moveWindow('test_bb8_pred', 230, 670)
            cv2.moveWindow('test_bb8_gt', 430, 670)
            cv2.waitKey(1)

        file_index = self.epoch.eval()
        if not os.path.exists(self.saver_log_folder + "render"):
            os.makedirs(self.saver_log_folder + "render")
        file = self.saver_log_folder + "render/result_" + str(file_index) + ".png"
        file1 = self.saver_log_folder + "render/gt_mask_" + str(file_index) + ".png"
        file2 = self.saver_log_folder + "render/result_rgb_" + str(file_index) + ".png"
        file3 = self.saver_log_folder + "render/result_bb8_" + str(file_index) + ".png"
        file4 = self.saver_log_folder + "render/result_bb8_pred_" + str(file_index) + ".png"
        file5 = self.saver_log_folder + "render/gt_bb8_" + str(file_index) + ".png"
        cv2.imwrite(file, test_img)
        cv2.imwrite(file1, test_gt_mask)
        cv2.imwrite(file2, test_rgb_raw)
        cv2.imwrite(file3, test_rgb)
        cv2.imwrite(file4, test_rgb_bb8_p)
        cv2.imwrite(file5, test_rgb_bb8_gt)

        return output, train_prob, train_loss, train_bb8_loss


    # Test the trained network.
    def __test_step(self, sess, test_rgb, test_gt_mask, test_gt_bb8):
        # Test Stage 1
        test_predict, test_prob, test_loss = sess.run([self.seg_predictions,
                                                       self.probabilities,
                                                       self.cross_entropy_sum],
                                                      feed_dict={self.ph_rgb: test_rgb,
                                                                 self.ph_mask: test_gt_mask,
                                                                 self.keep_prob_seg: 1.0})

        # Generate the output from the first stage. These are the images with argmax(activations) applied
        output = sess.run(self.seg_predictions, feed_dict={self.ph_rgb: test_rgb,
                                                           self.ph_mask: test_gt_mask,
                                                           self.keep_prob_seg: 1.0})

        # Calculate bb8 loss.
        test_bb8_loss = sess.run(self.loss_bb8, feed_dict={self.ph_seg_pred: output,
                                                           self.ph_rgb: test_rgb,
                                                           self.ph_gt_bb8: test_gt_bb8,
                                                           self.keep_prob_pose_conv: 1.0,
                                                           self.keep_prob_pose_hidden: 1.0})

        return test_predict, test_prob, test_loss, test_bb8_loss


    # Set training parameters.
    def setParams(self, num_epochs, batch_size, test_size):
        """
        :param num_epochs: (int)
            Set the number of epochs to train. Note that is a relative number and not the global, already trained
            epoch. The number set will be added to the total number of epochs to train.
        :param batch_size: (int)
            The batch size for mini-batch training as int
        :param test_size: (int)
            The test size for testing. Note that the test size should be larger than the batch size.
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.test_size = test_size

    # Set a log file path and a log file name.
    # The solve logs the tensorflow checkpoints automatically each 10 epochs.
    # Set the path and the logfile using this function. The log_path is also used for
    # all the other output files. The solver will not log anything if no path is given.
    def setLogPathAndFile(self, log_path, log_file):
        """
        :param log_path: (str)
            A string with a relative or absolute path. Complete the log path with a /
        :param log_file: (str)
            A string containing the log file name.
        """
        self.saver_log_folder = log_path
        self.saver_log_file = log_file

    # Calculate the accuracy of all images as precision and recall values.
    # Only for the segmentation part.
    def __getAccuracy(self, Y_pm, Ypr, validation="", start_index=0):
        """
        :param Y_pm: (array) The ground truth data mask as array of size [N, width*height], target class only, object 1.
        :param Ypr: (array) The prediction, index aligned with the ground truth data [N, width, height, 1], object 1.
        :param validation: (str), set a filename. If the string length is > 0, the results will be written into this file.
        :param start_index: (int), for the file writer; a batch indices that indicates the number of the current batch.
        :return: the precision (float) and recall (float) values.
        """

        Ypr_invert = np.not_equal(Ypr, 1) # background 1, object 0
        pr = Ypr_invert.reshape([-1, self.imgWidth, self.imgHeight]).astype(float)
        pr = pr * 255

        Y_pm_invert = np.not_equal(Y_pm, 1) # background 1, object 0
        y = Y_pm_invert.reshape([-1, self.imgWidth, self.imgHeight]).astype(float)
        y = y * 255

        N = Y_pm.shape[0]  # size

        if len(validation) > 0 and start_index == 0:
            file = open(self.saver_log_folder + validation, "w")
            file_str = "idx,precision,recall,accuracy\n"
            file.write(file_str)
            file.close()

        recall = 0
        precision = 0

        imgsize = self.imgWidth * self.imgHeight
        accuracy = 0

        for i in range(N):
            pr0 = pr[i]
            y0 = y[i]
            this_recall = 0
            this_precision = 0
            this_accuracy = 0

            # number of positive examples
            relevant = np.sum(np.equal(y0, 0).astype(int))  # tp + fn
            tp_map = np.add(pr0, y0)  # all true positive end up as 0 after the addition.
            tp = np.sum(np.equal(tp_map, 0).astype(int))  # count all true positive, tp
            this_recall = (tp.astype(float) / (relevant.astype(float)+0.0001))  # recall = tp / (tp + fn)
            recall = recall + this_recall

            # get all true predicted = tp and fp
            pr_true = np.sum(np.equal(pr0, 0).astype(int))  # tp + fp
            this_precision = tp.astype(float) / (pr_true.astype(float)+0.0001) # precision = tp / (tp + fp)
            precision = precision + this_precision

            # accuracy = (tp + tn)/all pixels
            tp_acc_map = np.subtract(pr0, y0)  # all true positive and true negative end up as 0 after the subtraction.
            correct = 0
            correct = np.sum(np.equal(tp_acc_map, 0).astype(int))  # count all true positive and true negative
            this_accuracy = correct.astype(float) / float(imgsize)
            accuracy = accuracy + this_accuracy

            if len(validation) > 0:
                file = open(self.saver_log_folder + validation, "a")
                file_str = str(start_index + i) + "," + str(this_precision) + "," + str(this_recall) + "," + str(this_accuracy) + "\n"
                file.write(file_str)
                file.close()

        recall = recall / float(N)
        precision = precision / float(N)
        accuracy = accuracy / float(N)

        return precision, recall, accuracy

    # Show or hide all debug outputs.
    def showDebug(self, show_plot):
        """
        A debug window showing the predicted image mask and the RGB image
        shows the results of each batch and the test results after each epoch.
        True activates this feature, False deactives it.

        :param show_plot: (bool)
            True shows all debug outputs, False will hide them.
        """
        self.show_sample = show_plot