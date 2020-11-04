import numpy as np
import os
import os.path


class Datalog:

    _path = ""
    _filename = "datalog.csv"

    def StartLog(self, path):
        self._path = path

        self.Exist(self._path)

        file = open(self._path + self._filename, "w")
        file_str = "Epoch," \
                   "Train Seg Loss,Train Seg Precison,Train Seg Recall,Train Seg Accuracy,Train BB8 Loss," \
                   "Test Seg Loss,Test Seg Precision,Test Seg Recall,Test Seg Accuracy,Test BB8 Loss\n"
        file.write(file_str)
        file.close()

    def AddData(self, step, train_seg_loss, train_seg_precision, train_seg_recall, train_seg_accuracy, train_bb8_loss,
                test_seg_loss, test_seg_precision, test_seg_recall, test_seg_accuracy, test_bb8_loss):

        file_str = str(step) + "," + \
                   str(train_seg_loss) + "," + \
                   str(train_seg_precision) + "," + str(train_seg_recall) + "," + \
                   str(train_seg_accuracy) + "," + str(train_bb8_loss) + "," + \
                   str(test_seg_loss) + "," + \
                   str(test_seg_precision) + "," + str(test_seg_recall) + "," + \
                   str(test_seg_accuracy) + "," + str(test_bb8_loss) + "\n"

        file = open(self._path + self._filename, "a")
        file.write(file_str)
        file.close()

    def Exist(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)