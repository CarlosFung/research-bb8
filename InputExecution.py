import sys
sys.dont_write_bytecode = True

from Experiment import *

Exp = {}

Exp["train_dataset_folder"] = "./data/32x32/training"
Exp["test_dataset_folder"] = "./data/32x32/testing"
# Exp["train_dataset_folder"] = "./data/64x64/training"
# Exp["test_dataset_folder"] = "./data/64x64/testing"

Exp["solver"] = TrainRGB
Exp["model"] = Model_BB8
Exp["learning_rate"] = 0.001
Exp["num_iterations"] = 100
    # 200

Exp["train"] = False  # training
Exp["eval"] = False  # cross-validation; no need to set to True, because already do so during training.

Exp["test"] = False  # testing

Exp["quality_check"] = 200  # 0/200/1800 samples quality check (dataset with 2000 images, train:valid=9:1)
Exp["quality_check_dataset"] = "TEST"  #"TEST": Cross-validation Set, 200; "TRAIN": Training Set, 1800.
Exp["quality_check_class_name"] = "01"  # if "", use training dataset with all classes; ow, use testing dataset

Exp["debug_output"] = True

Exp["log_path"] = "./logs/BOP_32x32_10.30_2/"
Exp["log_file"] = "BOP_32x32_10.30_2"

# Keep empty to not restore a model / Change to "" if only train (train for the first time):
# Exp["restore_file"] = ""
# Use it like this when evaluating:
Exp["restore_file"] = "BOP_32x32_10.30_2-100.meta"
# Exp["restore_file"] = ""



# Exp["quat_used"] = True  # set true, if the dataset contains quaternions. Otherwise false.
# Exp["plot_title"] = "BOP 64x64 RGB, 10.22"
# Exp["label"] = "Experiment.py"


experiment = Experiment(Exp)
experiment.start()