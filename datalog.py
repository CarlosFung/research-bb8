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
        file_str = "epoch,seg_loss,seg_precision,seg_recall,seg_accuracy,bb8_loss\n"
        file.write(file_str)
        file.close()

    def AddData(self, step, seg_loss, seg_precision, seg_recall, seg_accuracy, bb8_loss):

        file_str = str(step) + "," + \
                   str(seg_loss) + "," + str(seg_precision) + "," + str(seg_recall) + "," + str(seg_accuracy) + "," +\
                   str(bb8_loss) + "\n"

        file = open(self._path + self._filename, "a")
        file.write(file_str)
        file.close()

    def Exist(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)