import sys, tf
import models.baseline as baseline_model

from data_mgmt.data_mgmt import new_dataset

# Data manager parameters
reshuffle = True if '--reshuffle' in sys.argv else False
dataset_tsv_file = sys.argv[-2]

# Model parameters
retrain = True if '--retrain' in sys.argv else False 
save = True if '--save' in sys.argv else False
training_set_ratio = sys.argv[-1]

# TODO: Change reshuffle parameter
if reshuffle:
    new_dataset(dataset_tsv_file, training_set_ratio)

baseline_model.train_and_test(retrain, save)
