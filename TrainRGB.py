import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import pickle
import os
from os import path
from Model_BB8 import *
from layers import *
from datalog import *
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

    # quality check
    samplescheck = 0
    samplescheck_class_name = "random"

    # placeholders
    ph_rgb = []  # input RGB data of size [N, height, width, 3]
    ph_seg_pred = []  # prediction output of the first network of size [N, height, width, 1]
    ph_mask = []  # ground truth mask of size [N, width*height, C]
    ph_gt_bb8 = []  # ground truth bounding box 8 corners of size [N, 16]
    ph_gt_bb8_pairs = []  # ground truth bounding box 8 corners of size [N, 8, 2]

    # Dropout ratio
    keep_prob_seg = []
    keep_prob_pose_conv = []
    keep_prob_pose_hidden = []

    # Batch normalization
    phase_train = []  # training stage or not

    # References to data
    # All those variables point to the data location
    train_rgb = []     # Training RGB data
    train_gt_mask = [] # Training ground truth mask
    train_gt_one_mask = []  # Training ground truth mask, one class only
    train_gt_one_mask_m = []  # Training ground truth mask, one class only, in matrix
    train_gt_bb8 = []  # Training ground truth bounding box 8 corners
    train_gt_pose = [] # Training ground truth pose

    test_rgb = []      # Testing RGB data
    test_gt_mask = []  # Testing ground truth mask
    test_gt_one_mask = []  # Testing ground truth mask, one class only
    test_gt_one_mask_m = []  # Testing ground truth mask, one class only, in matrix
    test_gt_bb8 = []   # Testing ground truth bounding box 8 corners
    test_gt_pose = []  # Testing ground truth pose

    # predictions output
    pre_logits = [] # output of stage 1, activations
    seg_predictions = [] # output of stage 1, class predictions.
    pre_bb8 = []  # pose output of stage 2
    bb8_predict = [] # pose output of stage 2, reshaped as point pairs

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

    mydatalog = 0

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
    def init_graph(self, img_height, img_width, restore_from_file = ""):
        self.restore_from_file = restore_from_file

        self.epoch = tf.Variable(0, name='epoche', trainable=False)

        self.imgWidth = img_width
        self.imgHeight = img_height

        # placeholders
        # the input image of size
        # [128, 128, 3]
        self.ph_rgb = tf.placeholder("float", [None, img_height, img_width, 3])

        # predictions from the first network, input for the second network.
        self.ph_seg_pred = tf.placeholder("float", [None, img_height, img_width, 1])

        # vector for the output data.
        # The network generates an output layer for each class.
        # 16384 = 128 * 128
        # [N, width*height, C]
        self.ph_mask = tf.placeholder("float", [None, img_width * img_height, self.number_classes], name="gt_mask")

        # the bb8 ground truth
        self.ph_gt_bb8 = tf.placeholder("float", [None, self.num_BB8_outputs], name="gt_bb8")
        self.ph_gt_bb8_pairs = tf.placeholder("float", [None, self.num_BB8_outputs/2, 2], name="gt_bb8_pairs")

        # Dropout ratio placeholder
        self.keep_prob_seg = tf.placeholder("float", name="keep_prob_seg")
        self.keep_prob_pose_conv = tf.placeholder("float", name="keep_prob_pose_conv")
        self.keep_prob_pose_hidden = tf.placeholder("float", name="keep_prob_pose_hidden")

        # Batch normalization. Indicate if it is training stage or not
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')

        # This solver will create its own instance of the model
        self.model = self.model_cls(self.number_classes, self.num_BB8_outputs)

        # [N,128x128,16]       [N,128,128,1]   [N,16], i.e.(x1,y1,x2,y2,...,x8,y8)
        self.pre_logits, self.seg_predictions, self.pre_bb8 = \
            self.model.inference_op(self.ph_rgb, self.ph_seg_pred, self.keep_prob_seg,
                                    self.keep_prob_pose_conv, self.keep_prob_pose_hidden,
                                    self.phase_train)

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
        self.train_rgb = train_rgb   # Training RGB data [N, height, width, 3]
        self.train_gt_mask = train_gt_mask # Training ground truth mask [N, width*height, C]
        self.train_gt_one_mask = train_gt_one_mask # Training ground truth mask [N, width*height]
        self.train_gt_one_mask_m = train_gt_one_mask.reshape([-1, self.imgHeight, self.imgWidth, 1])  # Training ground truth mask [N, height, width, 1]
        self.train_gt_bb8 = train_gt_bb8  # Training ground truth bounding box 8 corners [N, 16]
        self.train_gt_pose = train_gt_pose  # Training ground truth pose [x, y, z, qx, qy, qz, qw]

        self.test_rgb = test_rgb   # Testing RGB data [N, height, width, 3]
        self.test_gt_mask = test_gt_mask # Testing ground truth mask [N, width*height, C]
        self.test_gt_one_mask = test_gt_one_mask # Testing ground truth mask [N, width*height]
        self.test_gt_one_mask_m = test_gt_one_mask.reshape([-1, self.imgHeight, self.imgWidth, 1])  # Testing ground truth mask [N, height, width, 1]
        self.test_gt_bb8 = test_gt_bb8  # Testing ground truth bounding box 8 corners [N, 16]
        self.test_gt_pose = test_gt_pose  # Testing ground truth pose [x, y, z, qx, qy, qz, qw]

        # invokes training
        self.__start_train()

    # Start the evaluation of the current model.
    def eval(self, rtest_rgb, rtest_mask, rtest_pm, rtest_bb8, rtest_pose):
        self.__start_eval(rtest_rgb, rtest_mask, rtest_pm, rtest_bb8, rtest_pose)

    # Start the quality check of the current model at current epoch.
    def qualityCheck(self, train_rgb, train_gt_mask, train_gt_one_mask, train_gt_bb8):
        self.__start_check(train_rgb, train_gt_mask, train_gt_one_mask, train_gt_bb8)


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

        self.bb8_predict = tf.reshape(self.pre_bb8, [-1, int(self.num_BB8_outputs/2), 2])
        self.ph_gt_bb8_pairs = tf.reshape(self.ph_gt_bb8, [-1, int(self.num_BB8_outputs/2), 2])
        self.loss_bb8 = tf.reduce_mean(tf.norm(tf.subtract(self.ph_gt_bb8_pairs, self.bb8_predict), axis=2))

        # Training with RMSProp optimizer.
        self.optimizer2 = tf.train.RMSPropOptimizer(0.0005, 0.9)

        # reduce the gradients.
        gradients2 = self.optimizer2.compute_gradients(loss=self.loss_bb8)

        # The training step.
        self.train_bb8 = self.optimizer2.apply_gradients(grads_and_vars=gradients2)

        # self.train_bb8 = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.loss_bb8)

        return self.train_seg, self.prediction, self.probabilities, self.train_bb8


    # Start the training procedure.
    def __start_train(self):
        # -------------------------------------------------------------------------------------
        # Init file writer
        self.mydatalog = Datalog()
        self.mydatalog.StartLog(self.saver_log_folder)

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

            # i: 0~num_epochs-1
            for i in range(self.num_epochs):
                # Count the steps manually
                self.step = i + start_idx  # step: 1~num_epochs, if this is a new model
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
                        self.__train_step(sess, self.step, self.train_rgb[train_indices],
                                          self.train_gt_mask[train_indices],
                                          self.train_gt_one_mask_m[train_indices],
                                          self.train_gt_bb8[train_indices],
                                          1.0, 0.9, 0.9, True)

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
                # Test accuracy/Cross-validation
                test_indices = np.arange(len(self.test_rgb))
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:self.test_size]
                test_predict, test_prob, test_loss, test_bb8_loss = \
                    self.__test_step(sess, self.step, self.test_rgb[test_indices],
                                     self.test_gt_mask[test_indices],
                                     self.test_gt_one_mask_m[test_indices],
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
                if self.step % 10 == 0:
                    self.saver.save(sess, self.saver_log_folder + self.saver_log_file, global_step=self.step)
                    print("Saved at step ", self.step)

                self.mydatalog.AddData(self.step,
                                       train_loss_avg / (self.imgWidth * self.imgHeight),
                                       train_precision_avg, train_recall_avg, train_accuracy_avg,
                                       train_bb8_loss_avg,
                                       test_loss / (self.imgWidth * self.imgHeight),
                                       test_precision, test_recall, test_accuracy,
                                       test_bb8_loss)

            # save the last step
            # self.saver.save(sess, self.saver_log_folder + self.saver_log_file, global_step=self.step)

            # keeps the file so that the evaluation can restore the last model.
            self.restore_from_file = self.saver_log_file + "-" + str(self.step) + ".meta"


    # Start the network evaluation.
    def __start_eval(self, rtest_rgb, rtest_mask, rtest_pm, rtest_bb8, rtest_pose):
        """
        :param rtest_rgb: (array) Testing RGB data [N, height, width, 3]
        :param rtest_mask: (array) Testing ground truth mask [N, width*height, C]
        :param rtest_pm: (array) Testing target object ground truth mask [N, width*height]
        :param rtest_bb8: (array) Testing ground truth bounding box 8 corners [N, 16]
        :param rtest_pose: (array) Testing ground truth pose [x, y, z, qx, qy, qz, qw]
        """

        # Check if a model has been restored.
        if len(self.restore_from_file) == 0:
            print("ERROR - Test/Validation mode requires a restored model")
            return

        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # run the test session
        with tf.Session() as sess:
            sess.run(initop)

            # Restore the graph
            saver = tf.train.import_meta_graph(self.saver_log_folder + self.restore_from_file)
            saver.restore(sess, tf.train.latest_checkpoint(self.saver_log_folder))
            curr_epoch = sess.run('epoche:0')
            print("Model restored at epoch ", curr_epoch)

            print("Start testing/validation..........")
            print("Num test samples: ", int(rtest_rgb.shape[0]), ".")

            # split the evaluation batch in small chunks
            # Note that the evaluation images are not shuffled to keep them aligned with the input images
            # test_size = self.test_size
            test_size = int(rtest_rgb.shape[0] / 9)
            validation_batch = zip(range(0, len(rtest_rgb), test_size), range(test_size, len(rtest_rgb) + 1, test_size))

            # resize the pure mask
            rtest_pm_m = rtest_pm.reshape([-1, self.imgHeight, self.imgWidth, 1])  # Testing ground truth mask [N, height, width, 1]

            # run the batch
            rtest_cnt = 0
            rtest_precision_sum = 0
            rtest_recall_sum = 0
            rtest_accuracy_sum = 0
            rtest_loss_sum = 0
            rtest_bb8_loss_sum = 0
            for start, end in validation_batch:
                rtest_predict, rtest_prob, rtest_loss, rtest_bb8_loss = \
                    self.__test_step(sess, curr_epoch, rtest_rgb[start:end],
                                     rtest_mask[start:end],
                                     rtest_pm_m[start:end],
                                     rtest_bb8[start:end])
                rtest_precision, rtest_recall, rtest_accuracy = self.__getAccuracy(rtest_pm[start:end], rtest_predict,
                                                                                   "valid_accuracy_" + str(rtest_cnt)
                                                                                   + ".csv")

                rtest_precision_sum = rtest_precision_sum + rtest_precision
                rtest_recall_sum = rtest_recall_sum + rtest_recall
                rtest_accuracy_sum = rtest_accuracy_sum + rtest_accuracy
                rtest_loss_sum = rtest_loss_sum + rtest_loss
                rtest_bb8_loss_sum = rtest_bb8_loss_sum + rtest_bb8_loss
                print("Batch,", rtest_cnt,
                      ",Valid Seg Loss,", rtest_loss / (self.imgWidth * self.imgHeight),
                      ",Valid Seg Precison,", rtest_precision,
                      ",Valid Seg Recall,", rtest_recall,
                      ",Valid Seg Accuracy,", rtest_accuracy,
                      ",Valid BB8 Loss,", rtest_bb8_loss)
                rtest_cnt = rtest_cnt + 1

            rtest_precision_avg = rtest_precision_sum / rtest_cnt
            rtest_recall_avg = rtest_recall_sum / rtest_cnt
            rtest_accuracy_avg = rtest_accuracy_sum / rtest_cnt
            rtest_loss_avg = rtest_loss_sum / rtest_cnt
            rtest_bb8_loss_avg = rtest_bb8_loss_sum / rtest_cnt

            print("Final results:\n",
                  "Valid Seg Loss,", rtest_loss_avg / (self.imgWidth * self.imgHeight),
                  ",Valid Seg Precison,", rtest_precision_avg,
                  ",Valid Seg Recall,", rtest_recall_avg,
                  ",Valid Seg Accuracy,", rtest_accuracy_avg,
                  ",Valid BB8 Loss,", rtest_bb8_loss_avg)

    # Start the network evaluation.
    def __start_check(self, qc_rgb, qc_gt_mask, qc_gt_one_mask, qc_gt_bb8):
        """
        :param qc_rgb: (array) Testing RGB data [N, height, width, 3]
        :param qc_gt_mask: (array) Testing ground truth mask [N, width*height, C]
        :param qc_gt_one_mask: (array) Testing target object ground truth mask [N, width*height]
        :param qc_gt_bb8: (array) Testing ground truth bounding box 8 corners [N, 16]
        """

        # Check if a model has been restored.
        if len(self.restore_from_file) == 0:
            print("ERROR - Test/Validation mode requires a restored model")
            return

        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # run the test session
        with tf.Session() as sess:
            sess.run(initop)

            # Restore the graph
            saver = tf.train.import_meta_graph(self.saver_log_folder + self.restore_from_file)
            saver.restore(sess, tf.train.latest_checkpoint(self.saver_log_folder))
            curr_epoch = sess.run('epoche:0')
            print("Model restored at epoch ", curr_epoch)

            print("Start quality check..........")
            print("Mode: ", self.samplescheck_class_name, "..........")
            print("Num quality check samples: ", self.samplescheck, ".")

            if not os.path.exists(self.saver_log_folder + "quality_check/" + self.samplescheck_class_name):
                os.makedirs(self.saver_log_folder + "quality_check/" + self.samplescheck_class_name)
            file = open(self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/qc_datalog.csv", "w")
            file_str = "Sample No.,QC Seg Loss,QC Seg Precison,QC Seg Recall,QC Seg Accuracy,QC BB8 Loss,QC BB8 Accuracy\n"
            file.write(file_str)
            file.close()

            # shuffle the indices
            indices = list(range(0, len(qc_rgb)))
            shuffled = np.random.permutation(indices)

            # resize the pure mask
            qc_gt_one_mask_m = qc_gt_one_mask.reshape([-1, self.imgHeight, self.imgWidth, 1])  # Testing ground truth mask [N, height, width, 1]

            qc_precision_sum = 0
            qc_recall_sum = 0
            qc_accuracy_sum = 0
            qc_loss_sum = 0
            qc_bb8_loss_sum = 0
            qc_bb8_accurate_count = 0
            # run the batch
            for idx in range(self.samplescheck):
                check_index = shuffled[idx]
                qc_predict, qc_prob, qc_loss, qc_bb8_loss = \
                    self.__qc_step(sess, curr_epoch, idx, qc_rgb[check_index],
                                   qc_gt_mask[check_index], qc_gt_one_mask_m[check_index],
                                   qc_gt_bb8[check_index])
                qc_precision, qc_recall, qc_accuracy = self.__getAccuracy(qc_gt_one_mask[check_index],
                                                                          qc_predict)
                if qc_bb8_loss < 7.0710678:
                    qc_bb8_accurate_count = qc_bb8_accurate_count + 1
                qc_precision_sum = qc_precision_sum + qc_precision
                qc_recall_sum = qc_recall_sum + qc_recall
                qc_accuracy_sum = qc_accuracy_sum + qc_accuracy
                qc_loss_sum = qc_loss_sum + qc_loss
                qc_bb8_loss_sum = qc_bb8_loss_sum + qc_bb8_loss
                print("Sample No.,", idx,
                      ",QC Seg Loss,", qc_loss / (self.imgWidth * self.imgHeight),
                      ",QC Seg Precison,", qc_precision,
                      ",QC Seg Recall,", qc_recall,
                      ",QC Seg Accuracy,", qc_accuracy,
                      ",QC BB8 Loss,", qc_bb8_loss,
                      ",QC BB8 AccurateCnt", qc_bb8_accurate_count)
                file_str = str(idx) + "," + \
                           str(qc_loss / (self.imgWidth * self.imgHeight)) + "," + \
                           str(qc_precision) + "," + \
                           str(qc_recall) + "," + \
                           str(qc_accuracy) + "," + \
                           str(qc_bb8_loss) + "," + \
                           str(qc_bb8_accurate_count) + "\n"
                file = open(self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/qc_datalog.csv", "a")
                file.write(file_str)
                file.close()

            qc_precision_avg = qc_precision_sum / self.samplescheck
            qc_recall_avg = qc_recall_sum / self.samplescheck
            qc_accuracy_avg = qc_accuracy_sum / self.samplescheck
            qc_loss_avg = qc_loss_sum / self.samplescheck
            qc_bb8_loss_avg = qc_bb8_loss_sum / self.samplescheck
            qc_bb8_accuracy = qc_bb8_accurate_count / self.samplescheck

            print("Mode " + self.samplescheck_class_name + ",", self.samplescheck,
                  ",QC Seg Loss,", qc_loss_avg / (self.imgWidth * self.imgHeight),
                  ",QC Seg Precison,", qc_precision_avg,
                  ",QC Seg Recall,", qc_recall_avg,
                  ",QC Seg Accuracy,", qc_accuracy_avg,
                  ",QC BB8 Loss,", qc_bb8_loss_avg,
                  ",QC BB8 Accuracy,", qc_bb8_accuracy)

            file_str = self.samplescheck_class_name + "," + \
                       str(qc_loss_avg / (self.imgWidth * self.imgHeight)) + "," + \
                       str(qc_precision_avg) + "," + \
                       str(qc_recall_avg) + "," + \
                       str(qc_accuracy_avg) + "," + \
                       str(qc_bb8_loss_avg) + "," + \
                       str(qc_bb8_accuracy) + "\n"
            file = open(self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/qc_datalog.csv", "a")
            file.write(file_str)
            file.close()


    # Execute one training step for the entire graph
    def __train_step(self, sess, epoch_num, rgb_batch, mask_batch, pm_m_batch, bb8_batch,
                     keep_prob_seg, keep_prob_pose_conv, keep_prob_pose_hidden, phase_train):
        # Train the first stage.
        sess.run(self.train_seg, feed_dict={self.ph_rgb: rgb_batch,
                                            self.ph_mask: mask_batch,
                                            self.keep_prob_seg: keep_prob_seg,
                                            self.phase_train: phase_train})
        train_prob, train_loss = sess.run([self.probabilities,
                                           self.cross_entropy_sum],
                                          feed_dict={self.ph_rgb: rgb_batch,
                                                     self.ph_mask: mask_batch,
                                                     self.keep_prob_seg: keep_prob_seg,
                                                     self.phase_train: phase_train})

        # Generate the output from the first stage - predicted mask.
        output = sess.run(self.seg_predictions, feed_dict={self.ph_rgb: rgb_batch,
                                                           self.ph_mask: mask_batch,
                                                           self.keep_prob_seg: keep_prob_seg,
                                                           self.phase_train: phase_train})

        # Train the second stage.
        if epoch_num > 30:
            sess.run(self.train_bb8, feed_dict={self.ph_seg_pred: output,
                                                self.ph_rgb: rgb_batch,
                                                self.ph_gt_bb8: bb8_batch,
                                                self.keep_prob_pose_conv: keep_prob_pose_conv,
                                                self.keep_prob_pose_hidden: keep_prob_pose_hidden,
                                                self.phase_train: phase_train})
            # Calculate bb8 loss.
            train_bb8_loss = sess.run(self.loss_bb8, feed_dict={self.ph_seg_pred: output,
                                                                self.ph_rgb: rgb_batch,
                                                                self.ph_gt_bb8: bb8_batch,
                                                                self.keep_prob_pose_conv: keep_prob_pose_conv,
                                                                self.keep_prob_pose_hidden: keep_prob_pose_hidden,
                                                                self.phase_train: phase_train})

            # Generate the output from the second stage - predicted bb8.
            output_bb8 = sess.run(self.pre_bb8, feed_dict={self.ph_seg_pred: output,
                                                           self.ph_rgb: rgb_batch,
                                                           self.keep_prob_pose_conv: keep_prob_pose_conv,
                                                           self.keep_prob_pose_hidden: keep_prob_pose_hidden,
                                                           self.phase_train: phase_train})
        else:  # epoch_num <= 30:
            sess.run(self.train_bb8, feed_dict={self.ph_seg_pred: pm_m_batch,
                                                self.ph_rgb: rgb_batch,
                                                self.ph_gt_bb8: bb8_batch,
                                                self.keep_prob_pose_conv: keep_prob_pose_conv,
                                                self.keep_prob_pose_hidden: keep_prob_pose_hidden,
                                                self.phase_train: phase_train})
            # Calculate bb8 loss.
            train_bb8_loss = sess.run(self.loss_bb8, feed_dict={self.ph_seg_pred: pm_m_batch,
                                                                self.ph_rgb: rgb_batch,
                                                                self.ph_gt_bb8: bb8_batch,
                                                                self.keep_prob_pose_conv: keep_prob_pose_conv,
                                                                self.keep_prob_pose_hidden: keep_prob_pose_hidden,
                                                                self.phase_train: phase_train})

            # Generate the output from the second stage - predicted bb8.
            output_bb8 = sess.run(self.pre_bb8, feed_dict={self.ph_seg_pred: pm_m_batch,
                                                           self.ph_rgb: rgb_batch,
                                                           self.keep_prob_pose_conv: keep_prob_pose_conv,
                                                           self.keep_prob_pose_hidden: keep_prob_pose_hidden,
                                                           self.phase_train: phase_train})

        # Shows an RGB image, the segmentation output and its predicted bb8.
        if self.show_sample:
            test_img = output[0] * 255
            test_gt_mask = pm_m_batch[0].copy()
            test_gt_mask = np.reshape(test_gt_mask, newshape=(self.imgHeight, self.imgWidth)) * 255
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
    def __test_step(self, sess, epoch_num, test_rgb, test_gt_mask, test_gt_pm_m, test_gt_bb8):
        # Test Stage 1
        test_predict, test_prob, test_loss = sess.run([self.seg_predictions,
                                                       self.probabilities,
                                                       self.cross_entropy_sum],
                                                      feed_dict={self.ph_rgb: test_rgb,
                                                                 self.ph_mask: test_gt_mask,
                                                                 self.keep_prob_seg: 1.0,
                                                                 self.phase_train: False})

        # Generate the output from the first stage. These are the images with argmax(activations) applied
        output = sess.run(self.seg_predictions, feed_dict={self.ph_rgb: test_rgb,
                                                           self.ph_mask: test_gt_mask,
                                                           self.keep_prob_seg: 1.0,
                                                           self.phase_train: False})

        # Calculate bb8 loss.
        if epoch_num > 30:
            test_bb8_loss = sess.run(self.loss_bb8, feed_dict={self.ph_seg_pred: output,
                                                               self.ph_rgb: test_rgb,
                                                               self.ph_gt_bb8: test_gt_bb8,
                                                               self.keep_prob_pose_conv: 1.0,
                                                               self.keep_prob_pose_hidden: 1.0,
                                                               self.phase_train: False})
        else:  # epoch_num <= 30:
            test_bb8_loss = sess.run(self.loss_bb8, feed_dict={self.ph_seg_pred: test_gt_pm_m,
                                                               self.ph_rgb: test_rgb,
                                                               self.ph_gt_bb8: test_gt_bb8,
                                                               self.keep_prob_pose_conv: 1.0,
                                                               self.keep_prob_pose_hidden: 1.0,
                                                               self.phase_train: False})

        return test_predict, test_prob, test_loss, test_bb8_loss


    # Quality check the trained network.
    def __qc_step(self, sess, epoch_num, qc_idx, qc_rgb, qc_gt_mask, qc_gt_pm_m, qc_gt_bb8):
        """
        :param qc_rgb: (array) Testing RGB data [height, width, 3]
        :param qc_gt_mask: (array) Testing ground truth mask [width*height, C]
        :param qc_gt_pm_m: (array) Testing target object ground truth mask [height, width]
        :param qc_gt_bb8: (array) Testing ground truth bounding box 8 corners [16]
        """
        qc_rgb = np.reshape(qc_rgb, newshape=(1, qc_rgb.shape[0], qc_rgb.shape[1], qc_rgb.shape[2]))
        qc_gt_mask = np.reshape(qc_gt_mask, newshape=(1, qc_gt_mask.shape[0], qc_gt_mask.shape[1]))
        qc_gt_pm_m = np.reshape(qc_gt_pm_m, newshape=(1, qc_gt_pm_m.shape[0], qc_gt_pm_m.shape[1]))
        qc_gt_bb8 = np.reshape(qc_gt_bb8, newshape=(1, qc_gt_bb8.shape[0]))

        # Test Stage 1
        qc_predict, qc_prob, qc_loss = sess.run([self.seg_predictions,
                                                 self.probabilities,
                                                 self.cross_entropy_sum],
                                                feed_dict={self.ph_rgb: qc_rgb,
                                                           self.ph_mask: qc_gt_mask,
                                                           self.keep_prob_seg: 1.0,
                                                           self.phase_train: False})

        # Generate the output from the first stage. These are the images with argmax(activations) applied
        output = sess.run(self.seg_predictions, feed_dict={self.ph_rgb: qc_rgb,
                                                           self.ph_mask: qc_gt_mask,
                                                           self.keep_prob_seg: 1.0,
                                                           self.phase_train: False})

        # Calculate bb8 loss.
        if epoch_num > 30:
            qc_bb8_loss = sess.run(self.loss_bb8, feed_dict={self.ph_seg_pred: output,
                                                             self.ph_rgb: qc_rgb,
                                                             self.ph_gt_bb8: qc_gt_bb8,
                                                             self.keep_prob_pose_conv: 1.0,
                                                             self.keep_prob_pose_hidden: 1.0,
                                                             self.phase_train: False})

            # Generate the output from the second stage - predicted bb8.
            output_bb8 = sess.run(self.pre_bb8, feed_dict={self.ph_seg_pred: output,
                                                           self.ph_rgb: qc_rgb,
                                                           self.keep_prob_pose_conv: 1.0,
                                                           self.keep_prob_pose_hidden: 1.0,
                                                           self.phase_train: False})

        else:  # epoch_num <= 30:
            qc_bb8_loss = sess.run(self.loss_bb8, feed_dict={self.ph_seg_pred: qc_gt_pm_m,
                                                             self.ph_rgb: qc_rgb,
                                                             self.ph_gt_bb8: qc_gt_bb8,
                                                             self.keep_prob_pose_conv: 1.0,
                                                             self.keep_prob_pose_hidden: 1.0,
                                                             self.phase_train: False})

            # Generate the output from the second stage - predicted bb8.
            output_bb8 = sess.run(self.pre_bb8, feed_dict={self.ph_seg_pred: qc_gt_pm_m,
                                                           self.ph_rgb: qc_rgb,
                                                           self.keep_prob_pose_conv: 1.0,
                                                           self.keep_prob_pose_hidden: 1.0,
                                                           self.phase_train: False})

        # Shows an RGB image, the segmentation output and its predicted bb8.
        if self.show_sample:
            qc_seg = output[0] * 255
            qc_mask = qc_gt_pm_m[0].copy() * 255
            qc_rgb_raw = qc_rgb[0].copy()
            qc_rgb_crop = qc_rgb[0].copy()
            qc_rgb_crop = np.multiply(qc_rgb_crop, output[0])
            qc_rgb_bb8_mixed = qc_rgb[0].copy()
            qc_rgb_bb8_p = qc_rgb[0].copy()
            qc_rgb_bb8_gt = qc_rgb[0].copy()
            qc_bb8 = np.array(output_bb8[0], dtype=np.int32)
            qc_bb8_gt = np.array(qc_gt_bb8[0], dtype=np.int32)

            p1 = (qc_bb8[0], qc_bb8[1])
            p2 = (qc_bb8[2], qc_bb8[3])
            p3 = (qc_bb8[4], qc_bb8[5])
            p4 = (qc_bb8[6], qc_bb8[7])
            p5 = (qc_bb8[8], qc_bb8[9])
            p6 = (qc_bb8[10], qc_bb8[11])
            p7 = (qc_bb8[12], qc_bb8[13])
            p8 = (qc_bb8[14], qc_bb8[15])
            p1gt = (qc_bb8_gt[0], qc_bb8_gt[1])
            p2gt = (qc_bb8_gt[2], qc_bb8_gt[3])
            p3gt = (qc_bb8_gt[4], qc_bb8_gt[5])
            p4gt = (qc_bb8_gt[6], qc_bb8_gt[7])
            p5gt = (qc_bb8_gt[8], qc_bb8_gt[9])
            p6gt = (qc_bb8_gt[10], qc_bb8_gt[11])
            p7gt = (qc_bb8_gt[12], qc_bb8_gt[13])
            p8gt = (qc_bb8_gt[14], qc_bb8_gt[15])

            blue = [255, 0, 0]
            green = [0, 255, 0]
            red = [0, 0, 255]
            yellow = [0, 255, 255]
            magenta = [255, 0, 255]
            cyan = [255, 255, 0]
            pink = [175, 175, 255]
            orange = [0, 127, 255]

            # cv2.circle(image, (x, y), radius, (B,G,R), thickness)
            cv2.circle(qc_rgb_bb8_mixed, p1, 1, blue)
            cv2.circle(qc_rgb_bb8_mixed, p2, 1, green)
            cv2.circle(qc_rgb_bb8_mixed, p3, 1, red)
            cv2.circle(qc_rgb_bb8_mixed, p4, 1, yellow)
            cv2.circle(qc_rgb_bb8_mixed, p5, 1, magenta)
            cv2.circle(qc_rgb_bb8_mixed, p6, 1, cyan)
            cv2.circle(qc_rgb_bb8_mixed, p7, 1, pink)
            cv2.circle(qc_rgb_bb8_mixed, p8, 1, orange)
            cv2.circle(qc_rgb_bb8_mixed, p1gt, 1, blue, -1)
            cv2.circle(qc_rgb_bb8_mixed, p2gt, 1, green, -1)
            cv2.circle(qc_rgb_bb8_mixed, p3gt, 1, red, -1)
            cv2.circle(qc_rgb_bb8_mixed, p4gt, 1, yellow, -1)
            cv2.circle(qc_rgb_bb8_mixed, p5gt, 1, magenta, -1)
            cv2.circle(qc_rgb_bb8_mixed, p6gt, 1, cyan, -1)
            cv2.circle(qc_rgb_bb8_mixed, p7gt, 1, pink, -1)
            cv2.circle(qc_rgb_bb8_mixed, p8gt, 1, orange, -1)

            cv2.circle(qc_rgb_bb8_p, p1, 1, blue)
            cv2.circle(qc_rgb_bb8_p, p2, 1, green)
            cv2.circle(qc_rgb_bb8_p, p3, 1, red)
            cv2.circle(qc_rgb_bb8_p, p4, 1, yellow)
            cv2.circle(qc_rgb_bb8_p, p5, 1, magenta)
            cv2.circle(qc_rgb_bb8_p, p6, 1, cyan)
            cv2.circle(qc_rgb_bb8_p, p7, 1, pink)
            cv2.circle(qc_rgb_bb8_p, p8, 1, orange)

            cv2.circle(qc_rgb_bb8_gt, p1gt, 1, blue, -1)
            cv2.circle(qc_rgb_bb8_gt, p2gt, 1, green, -1)
            cv2.circle(qc_rgb_bb8_gt, p3gt, 1, red, -1)
            cv2.circle(qc_rgb_bb8_gt, p4gt, 1, yellow, -1)
            cv2.circle(qc_rgb_bb8_gt, p5gt, 1, magenta, -1)
            cv2.circle(qc_rgb_bb8_gt, p6gt, 1, cyan, -1)
            cv2.circle(qc_rgb_bb8_gt, p7gt, 1, pink, -1)
            cv2.circle(qc_rgb_bb8_gt, p8gt, 1, orange, -1)

            cv2.imshow("qc_seg", qc_seg)
            cv2.imshow("qc_gt_mask", qc_mask)
            cv2.imshow("qc_rgb_crop", qc_rgb_crop)
            cv2.imshow("qc_rgb", qc_rgb_raw)
            cv2.imshow("qc_bb8_mixed", qc_rgb_bb8_mixed)
            cv2.imshow("qc_bb8_pred", qc_rgb_bb8_p)
            cv2.imshow("qc_bb8_gt", qc_rgb_bb8_gt)

            cv2.moveWindow('qc_seg', 30, 450)
            cv2.moveWindow('qc_gt_mask', 230, 450)
            cv2.moveWindow('qc_rgb_crop', 230, 560)
            cv2.moveWindow('qc_rgb', 30, 560)
            cv2.moveWindow('qc_bb8_mixed', 30, 670)
            cv2.moveWindow('qc_bb8_pred', 230, 670)
            cv2.moveWindow('qc_bb8_gt', 430, 670)
            cv2.waitKey(1)

        epoch_index = epoch_num
        file_index = qc_idx

        file = self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/result_seg_e" + str(epoch_index) + "_" + str(file_index) + ".png"
        file1 = self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/gt_mask_e" + str(epoch_index) + "_" + str(file_index) + ".png"
        file2 = self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/result_crop_e" + str(epoch_index) + "_" + str(file_index) + ".png"
        file3 = self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/rgb_e" + str(epoch_index) + "_" + str(file_index) + ".png"
        file4 = self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/result_bb8_mixed_e" + str(epoch_index) + "_" + str(file_index) + ".png"
        file5 = self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/result_bb8_pred_e" + str(epoch_index) + "_" + str(file_index) + ".png"
        file6 = self.saver_log_folder + "quality_check/" + self.samplescheck_class_name + "/gt_bb8_e" + str(epoch_index) + "_" + str(file_index) + ".png"
        cv2.imwrite(file, qc_seg)
        cv2.imwrite(file1, qc_mask)
        cv2.imwrite(file2, qc_rgb_crop)
        cv2.imwrite(file3, qc_rgb_raw)
        cv2.imwrite(file4, qc_rgb_bb8_mixed)
        cv2.imwrite(file5, qc_rgb_bb8_p)
        cv2.imwrite(file6, qc_rgb_bb8_gt)

        return qc_predict, qc_prob, qc_loss, qc_bb8_loss



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


    # Set quality check parameters.
    def setQualityCheck(self, num_samples):
        self.samplescheck = num_samples

    def setQCClass(self, class_name):
        self.samplescheck_class_name = class_name


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
        :param Ypr: (array) The prediction, index aligned with the ground truth data [N, height, width, 1], object 1.
        :param validation: (str), set a filename. If the string length is > 0, the results will be written into this file.
        :param start_index: (int), for the file writer; a batch indices that indicates the number of the current batch.
        :return: the precision (float) and recall (float) values.
        """

        Ypr_invert = np.not_equal(Ypr, 1) # background 1, object 0
        pr = Ypr_invert.reshape([-1, self.imgHeight, self.imgWidth]).astype(float)
        pr = pr * 255

        Y_pm_invert = np.not_equal(Y_pm, 1) # background 1, object 0
        y = Y_pm_invert.reshape([-1, self.imgHeight, self.imgWidth]).astype(float)
        y = y * 255

        N = y.shape[0]  # size

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