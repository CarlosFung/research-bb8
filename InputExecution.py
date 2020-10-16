from Experiment import *

Exp = {}

Exp["train_dataset_folder"] = "./data/pickles_32x32"
Exp["test_dataset"] = "./data/pickles_32x32/06.pickle"
Exp["solver"] = TrainRGB
Exp["model"] = Model_BB8
Exp["learning_rate"] = 0.001
Exp["num_iterations"] = 100
    # 200

Exp["train"] = False  # training
Exp["eval"] = False  # cross-validation; no need to set to True, because already do so during training.
Exp["test"] = True  # testing

Exp["debug_output"] = True

Exp["log_path"] = "./logs/BOP_32x32_10.14/"
Exp["log_file"] = "BOP_32x32_10.14"

# Keep empty to not restore a model / Change to "" if only train (train for the first time):
# Exp["restore_file"] = ""
# Use it like this when evaluating:
Exp["restore_file"] = "BOP_32x32_10.14-100.meta"
# Exp["restore_file"] = ""











Exp["quat_used"] = True  # set true, if the dataset contains quaternions. Otherwise false.
Exp["plot_title"] = "BOP 32x32 RGB, 10.14"
Exp["label"] = "Experiment.py"


experiment = Experiment(Exp)
experiment.start()