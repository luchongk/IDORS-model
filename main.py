import sys
import models.baseline as baseline_model
import numpy as np
import tensorflow as tf
from models.functional_model import FunctionalModel

from data_mgmt.data_mgmt import new_dataset, get_dataset, dataset_to_embeddings
from fasttext import load_model

from models.tf_model import train_step, test_step, TfModel

def labelProportion(trainLabels, testLabels):
    countYesTraining = 0
    countYesTest = 0

    countYesTraining = 0
    for label in trainLabels:
        if label == 1:
            countYesTraining += 1

    trainProportion = countYesTraining / len(trainLabels)

    countYesTest = 0
    for label in testLabels:
        if label == 1:
            countYesTest += 1
        
    testProportion = countYesTest / len(testLabels)

    allProportion = (countYesTraining + countYesTest) / (len(trainLabels) + len(testLabels))

    return (trainProportion, testProportion, allProportion)

# Data manager parameters
resplit = True if '--resplit' in sys.argv else False
dataset_tsv_file = sys.argv[-2]

# Model parameters
retrain = True if '--retrain' in sys.argv else False 
training_set_ratio = float(sys.argv[-1])
save = True if '--save' in sys.argv else False

# TODO: Change resplit parameter
if resplit:
    new_dataset(dataset_tsv_file, training_set_ratio)

training_dataset_text, test_dataset_text = get_dataset()

########## Train and Test Process ########## 
try:
    ft_model = load_model("baseline.bin")
except ValueError as err:
    print(err)
    print("Couldn't find a saved model, aborting...")
    exit(0)

training_dataset_embeddings = dataset_to_embeddings(training_dataset_text, ft_model)
training_dataset_labels = np.asarray([int(ex[1]) for ex in training_dataset_text])

test_dataset_embeddings = dataset_to_embeddings(test_dataset_text, ft_model)
test_dataset_labels = np.asarray([int(ex[1]) for ex in test_dataset_text])

#YES label proportions
trainProportion, testProportion, allProportion = labelProportion(list(training_dataset_labels), list(test_dataset_labels))

example_dim = training_dataset_embeddings[0].shape

model = FunctionalModel(example_dim)

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                                    restore_best_weights=True)

if retrain:
    model.fit(training_dataset_embeddings, training_dataset_labels,
                batch_size=64,
                epochs=400,
                validation_split=0.2,
                callbacks=[earlyStopping])
else:
    try:
        model.load_weights('tf_weights.h5')
    except ImportError as h5_err:
        print(h5_err)
        print("You need to install h5py to load a TensorFlow model, aborting...")
    except IOError as io_err:
        print(io_err)
        print("Couldn't find a saved TensorFlow model, aborting...")

if save:
    model.save_weights('tf_weights.h5')

loss, accuracy, precision, recall, auc = model.evaluate(test_dataset_embeddings, test_dataset_labels, verbose=2)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
print('Test precision:', precision)
print('Test recall:', recall)
print('Test AUC:', auc)
print('Test F-Score', 2 * (precision * recall) / (precision + recall))