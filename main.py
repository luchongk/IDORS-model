import sys, time, math
import numpy as np
import tensorflow as tf

from fasttext import load_model
from pprint import pprint
from pathlib import Path

from models.fasttext_model import baseline as baseline_model
from models.functional_model import FunctionalModel
from models.tf_model import TfModel
from data_mgmt.data_mgmt import new_dataset, get_dataset, dataset_to_embeddings

EPOCHS = 400

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
save = True if '--save' in sys.argv else False
functional = True if '--functional' in sys.argv else False
training_set_ratio = float(sys.argv[-1])

# Logging parameters
skipLogging = True if '--skip-logging' in sys.argv else False

# TODO: Change resplit parameter
if resplit:
    new_dataset(dataset_tsv_file, training_set_ratio)

training_dataset_text, test_dataset_text = get_dataset()

########## Train and Test Process ########## 
try:
    ft_model = load_model("models/fasttext_model/baseline.bin")
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

# Dimension of the tweet embeddings                                                
example_dim = training_dataset_embeddings[0].shape

training_dataset = None
test_dataset = None

if (functional):
    model = FunctionalModel(example_dim)
else:
    tf.keras.backend.set_floatx('float64')

    training_dataset = tf.data.Dataset.from_tensor_slices((training_dataset_embeddings, training_dataset_labels))
    training_dataset = training_dataset.shuffle(400).batch(64, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset_embeddings, test_dataset_labels))
    test_dataset = test_dataset.batch(64, drop_remainder=True)

    # Optimizer algorithm for training
    optimizer = tf.keras.optimizers.Adam()

    # Loss function
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Metrics that will measure loss and accuracy of the model over the training process
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_precision = tf.keras.metrics.Precision(name='train_precision', dtype='float32')
    train_recall = tf.keras.metrics.Recall(name='train_recall', dtype='float32')

    # Metrics that will measure loss and accuracy of the model over the testing process
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
    test_precision = tf.keras.metrics.Precision(name='test_precision', dtype='float32')
    test_recall = tf.keras.metrics.Recall(name='test_recall', dtype='float32')

    model = TfModel(example_dim,
                loss_object, 
                optimizer, 
                train_loss, 
                train_accuracy, 
                train_precision, 
                train_recall, 
                test_loss, 
                test_accuracy, 
                test_precision, 
                test_recall)

if retrain:
    if (functional):
        model.summary()
        
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                                    restore_best_weights=True)
        
        history = model.fit(training_dataset_embeddings, training_dataset_labels,
                    batch_size=64,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    callbacks=[earlyStopping])

        #TODO: Make an evaluate method for the subclassed model
        loss, accuracy, precision, recall, auc = model.evaluate(test_dataset_embeddings, test_dataset_labels, verbose=2)

        fscore = 2 * (precision * recall) / (precision + recall)

        template = '\n###### Test results ######\n\nTest Loss: {},\nTest Accuracy: {},\nTest Precision: {},\nTest Recall: {},\nTest AUC: {},\nTest F-Score: {}\n'
        
        print(template.format(loss, accuracy, precision, recall, auc, fscore))

        if not skipLogging:
            Path("runs").mkdir(parents=True, exist_ok=True)
            with open('runs/' + dataset_tsv_file.split('.tsv')[0] + str(math.trunc(time.time())), 'w') as logfile:
                logfile.write('Using dataset: ' + dataset_tsv_file + '\n')
                logfile.write('\n###### Model Summary ######\n\n'.format(EPOCHS))
                model.summary(print_fn=lambda x: logfile.write(x + '\n'))
                logfile.write(template.format(loss, accuracy, precision, recall, auc, fscore))
                logfile.write('\n###### Metrics history for {} epochs: ######\n\n'.format(len(history.epoch)))
                for epoch in history.epoch:
                    metricsHistory = history.history
                    logfile.write('Epoch {}: '.format(epoch + 1))
                    for key in metricsHistory.keys():
                        logfile.write('{}: {},'.format(key, metricsHistory[key][epoch]))
                        logfile.write(' ')
                    logfile.write('\n')
                logfile.write('\n##### Raw metrics history #####\n\n')
                pprint(metricsHistory, logfile)
    else:
        model.fit(training_dataset, test_dataset, EPOCHS)
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
        