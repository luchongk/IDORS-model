import sys
import models.baseline as baseline_model
import numpy as np
import tensorflow as tf

from data_mgmt.data_mgmt import new_dataset, get_dataset, dataset_to_embeddings
from fasttext import load_model

from models.tf_model import train_step, test_step, TfModel

# Data manager parameters
reshuffle = True if '--reshuffle' in sys.argv else False
dataset_tsv_file = sys.argv[-2]

# Model parameters
retrain = True if '--retrain' in sys.argv else False 
training_set_ratio = float(sys.argv[-1])
save = True if '--save' in sys.argv else False

# TODO: Change reshuffle parameter
if reshuffle:
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

## baseline_model.train_and_test(retrain, save)
## tf.compat.v1.disable_eager_execution() # Comment this if you want to debug the model
tf.keras.backend.set_floatx('float64')

training_dataset = tf.data.Dataset.from_tensor_slices((training_dataset_embeddings, training_dataset_labels))
training_dataset = training_dataset.shuffle(400).batch(64, drop_remainder=True)

test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset_embeddings, test_dataset_labels))
test_dataset = test_dataset.batch(64, drop_remainder=True)
                                                
example_dim = training_dataset_embeddings[0].shape

model = TfModel(example_dim)

# Optimizer algorithm for training
optimizer = tf.keras.optimizers.Adam()

# Metrics that will measure loss and accuracy of the model over the training process
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
train_precision = tf.keras.metrics.Precision(name='train_precision')
train_recall = tf.keras.metrics.Recall(name='train_recall')

# Metrics that will measure loss and accuracy of the model over the testing process
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
test_precision = tf.keras.metrics.Precision(name='test_precision')
test_recall = tf.keras.metrics.Recall(name='test_recall')

# Loss function
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

EPOCHS = 100

if not retrain:
    try:
        model.load_weights('tf_weights.h5')
    except ImportError as h5_err:
        print(h5_err)
        print("You need to install h5py to load a TensorFlow model, aborting...")
    except IOError as io_err:
        print(io_err)
        print("Couldn't find a saved TensorFlow model, aborting...")

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    if retrain:
        for images, labels in training_dataset:
            train_step(model, 
                       images, 
                       tf.reshape(labels, [64,1]), 
                       loss_object, 
                       optimizer, 
                       train_loss, 
                       train_accuracy, 
                       train_precision, 
                       train_recall)  

    if model:
        for test_images, test_labels in test_dataset:
            test_step(model, 
                      test_images, 
                      tf.reshape(test_labels, [64,1]), 
                      loss_object, 
                      test_loss, 
                      test_accuracy,
                      test_precision, 
                      test_recall)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Precision: {}, Recall: {}, Test Loss: {}, Test Accuracy: {}, Test Precision: {}, Test Recall: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        train_precision.result()*100,
                        train_recall.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100,
                        test_precision.result()*100,
                        test_recall.result()*100))

if save:
    model.save_weights('tf_weights.h5')       
