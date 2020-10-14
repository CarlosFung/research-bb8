from TrainRGB import *
from loaders import *
from quattool import *

# Universal class to run CNN model experiments.
class Experiment:
    _descrip = []
    _ready = False

    # Init the class with a description of the experiment.
    def __init__(self, description):
        self._descrip = description
        self._ready = True

    # Start the experiment
    def start(self):
        if self._ready == False:
            return

        # Load and prepare the data
        """
        The pickle contains a dict with the following labels:
            Xtr_rgb  - the training data set, rgb images of shape (num_images, width, height, channels = 3), uint8 [0,255]
            Ytr_mask - the training mask, each pixel in indicates the location of the object. [N, image h*w, c]
            Ytr_pm   - the training mask, target class only. [N, image h*w]
            Ytr_pose - the training poses of each object stored as (num_images, x, y, z, qx, qy, qz, qw)
            Ytr_bb8  - the training bounding box for each object stored as (num_images, x1, y1, x2, y2, ..., x8, y8)
            Xte_rgb  - the test data set, rgb images of shape (num_images, width, height, channels = 3)
            Yte_mask - the test mask, with each pixel indicating the location of the object. [N, image h*w, c]
            Yte_pm   - the test mask, target class only. [N, image h*w]
            Yte_pose - the test poses of each object stored as (num_images, x, y, z, qx, qy, qz, qw)
            Yte_bb8  - the test bounding box for each object stored as (num_images, x1, y1, x2, y2, ..., x8, y8)
        """

        #     0         1        2        3        4        5         6        7        8        9
        # [Xtr_rgb, Ytr_mask, Ytr_pm, Ytr_pose, Ytr_bb8, Xte_rgb, Yte_mask, Yte_pm, Yte_pose, Yte_bb8]
        #    RGB      Mask   Pure_mask  Pose      BB8      RGB      Mask   Pure_mask   Pose      BB8

        loaded_data = []
        loaded_files = os.listdir(self._descrip["train_dataset_folder"])
        for file in loaded_files:
            if not os.path.isdir(file):
                loaded_data_temp = prepare_data_RGB_6DoF(self._descrip["train_dataset_folder"]+"/"+file)
                if len(loaded_data) == 0:
                    loaded_data = loaded_data_temp
                else:
                    loaded_data[0] = np.concatenate((loaded_data[0], loaded_data_temp[0]), axis=0)
                    loaded_data[1] = np.concatenate((loaded_data[1], loaded_data_temp[1]), axis=0)
                    loaded_data[2] = np.concatenate((loaded_data[2], loaded_data_temp[2]), axis=0)
                    loaded_data[3] = np.concatenate((loaded_data[3], loaded_data_temp[3]), axis=0)
                    loaded_data[4] = np.concatenate((loaded_data[4], loaded_data_temp[4]), axis=0)
                    loaded_data[5] = np.concatenate((loaded_data[5], loaded_data_temp[5]), axis=0)
                    loaded_data[6] = np.concatenate((loaded_data[6], loaded_data_temp[6]), axis=0)
                    loaded_data[7] = np.concatenate((loaded_data[7], loaded_data_temp[7]), axis=0)
                    loaded_data[8] = np.concatenate((loaded_data[8], loaded_data_temp[8]), axis=0)
                    loaded_data[9] = np.concatenate((loaded_data[9], loaded_data_temp[9]), axis=0)


        # Synthetic data
        Xtr_rgb = loaded_data[0]
        Ytr_mask = loaded_data[1]
        Ytr_pm = loaded_data[2]

        Ytr_pose = loaded_data[3]
        Ytr_pose[:, 3:] = Quaternion.NormalizeList(Ytr_pose[:, 3:])  # normalizes the quaternions Ytr_pose[:, 3:]
        Ytr_bb8 = loaded_data[4]

        Xte_rgb = loaded_data[5]
        Yte_mask = loaded_data[6]
        Yte_pm = loaded_data[7]

        Yte_pose = loaded_data[8]
        Yte_pose[:, 3:] = Quaternion.NormalizeList(Yte_pose[:, 3:])  # normalizes the quaternions
        Yte_bb8 = loaded_data[9]

        # Init the network
        solver = self._descrip["solver"](self._descrip["model"], 16, 16, self._descrip["learning_rate"])
        solver.setParams(self._descrip["num_iterations"], 32, 32)
        solver.showDebug(self._descrip["debug_output"])
        solver.setLogPathAndFile(self._descrip["log_path"], self._descrip["log_file"])

        solver.init_graph(Xtr_rgb.shape[1], Xtr_rgb.shape[2], self._descrip["restore_file"])

        # Start to train the model.
        if self._descrip["train"]:
            solver.train(Xtr_rgb, Ytr_mask, Ytr_pm, Ytr_bb8, Ytr_pose,
                         Xte_rgb, Yte_mask, Yte_pm, Yte_bb8, Yte_pose)

        # Testing
        if self._descrip["eval"]:
            solver.eval(Xte_rgb, Yte_mask, Yte_pose)

        # Testing with a second test set.
        # i.e. Different datasets for training and validation.
        if self._descrip["test"] and len(self._descrip["test_dataset"]) > 0:
            loaded_eval_data = prepare_data_RGB_6DoF(self._descrip["test_dataset"])
            Xev_rgb = loaded_eval_data[0]
            Yev_mask = loaded_eval_data[1]
            Yev_pm = loaded_eval_data[2]
            Yev_pose = loaded_eval_data[3]
            Yev_pose[:, 3:] = Quaternion.NormalizeList(Yev_pose[:, 3:])  # normalizes the quaternions Yev_pose[:, 3:]
            Yev_bb8 = loaded_data[4]

            # swap colums for the quaternion from (x, y, z, w) -> (w, x, y, z)
            solver.eval(Xev_rgb, Yev_mask, Yev_pose)

        # Analyze the results
        # ????????????????????????????????????????????????????????????????????????????????????????????



