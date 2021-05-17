import sys
sys.dont_write_bytecode = True

from Experiment import *

Exp = {}

Exp["train_dataset_folder"] = "./data/32x32/training"
# Exp["test_dataset_folder"] = "./data/32x32/testing"
Exp["test_dataset_folder"] = ""  # No testing

# True: Handle one class
# False: Handle all classes; This is default
Exp["load_single_class"] = False
Exp["single_class_name"] = ""


Exp["solver"] = TrainRGB
Exp["model"] = Model_BB8
Exp["learning_rate"] = 0.001
Exp["num_iterations"] = 100 # 200

Exp["train"] = False  # training
Exp["eval"] = False  # cross-validation; no need to set to True, because already do so during training.

Exp["test"] = False  # testing

Exp["debug_output"] = True

Exp["log_path"] = "./logs/BOP_32x32_1.20_BN_0.9_0.9/"
Exp["log_file"] = "BOP_32x32_1.20_BN_0.9_0.9"

# Keep empty to not restore a model / Change to "" if only train (train for the first time):
# Exp["restore_file"] = ""
# Use it like this when evaluating:
Exp["restore_file"] = "BOP_32x32_1.20_BN_0.9_0.9-100.meta"





# Exp["quat_used"] = True  # set true, if the dataset contains quaternions. Otherwise false.
# Exp["plot_title"] = "BOP 64x64 RGB, 10.22"
# Exp["label"] = "Experiment.py"






Exp["quality_check"] = 200  # 0/200/1800 samples quality check (dataset with 2000 images, train:valid=9:1)

if Exp["quality_check"] == 0:
    experiment = Experiment(Exp)
    experiment.start()
else:
    # True: Use Exp["test_dataset_folder"]
    # False: Use Exp["train_dataset_folder"]
    Exp["quality_check_use_test_dataset"] = False

    # In the dataset we select, choose:
    # "TEST": Cross-validation Set size: 200;
    # "TRAIN": Training Set size: 1800.
    Exp["quality_check_dataset"] = "TEST"

    # True: Handle one class
    # False: Handle all classes
    Exp["load_single_class"] = True

    if Exp["load_single_class"]:
        idx = 14 # 1, 2, ..., 15
        if idx < 10:
            Exp["single_class_name"] = "0" + str(idx)
        else:
            Exp["single_class_name"] = str(idx)
        experiment = Experiment(Exp)
        experiment.start()
    else:
        Exp["single_class_name"] = ""
        experiment = Experiment(Exp)
        experiment.start()









