from __future__ import absolute_import, division, print_function, unicode_literals

import re, sys
import tensorflow as tf
import numpy as np

from fasttext import load_model

def get_dataset():
    parsing_regex = re.compile(r'^__label__(\d)\s{1}(.*)$')

    train_file = open('training_set.txt')
    test_file = open('test_set.txt')

    training_dataset = []
    test_dataset = []

    for line in train_file:
        match = re.match(parsing_regex, line)
        training_dataset.append((match[2], match[1]))

    for line in test_file:
        match = re.match(parsing_regex, line)
        test_dataset.append((match[2], match[1]))

    return training_dataset, test_dataset

def dataset_to_embeddings(dataset, ft_model):
    result = np.empty((len(dataset), ft_model.get_dimension()))

    for index, example in enumerate(dataset):
        result[index] = get_tweet_embedding(example[0], ft_model)

    return result
    
def get_tweet_embedding(tweet, ft_model):
    vec = np.repeat(0.0, ft_model.get_dimension())
    words = tweet.split()
    for word in words:
        vec += np.array(ft_model.get_word_vector(word))
    return np.divide(vec, len(words))

def train(training_dataset, example_dim, save):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=example_dim))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(training_dataset, epochs=3)

    if save:
        model.save('tf_model.h5')

    return model

def test(test_dataset, model):
    model.evaluate(test_dataset)

def train_and_test(training_dataset, test_dataset, example_dim, retrain, save):
    model = None
    
    if retrain:
        model = train(training_dataset, example_dim, save)
    else:
        try:
            model = tf.keras.models.load_model('tf_model.h5')
        except ImportError as h5_err:
            print(h5_err)
            print("You need to install h5py to load a TensorFlow model, aborting...")
        except IOError as io_err:
            print(io_err)
            print("Couldn't find a saved TensorFlow model, aborting...")
    
    if model:
        test(test_dataset, model)

if __name__ == "__main__":
    retrain = True if '--retrain' in sys.argv else False 
    save = True if '--save' in sys.argv else False

    training_dataset_text, test_dataset_text = get_dataset()
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

    training_dataset = tf.data.Dataset.from_tensor_slices((training_dataset_embeddings, training_dataset_labels))
    training_dataset = training_dataset.batch(64)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset_embeddings, test_dataset_labels))
    test_dataset = test_dataset.batch(64)
                                                    
    example_dim = training_dataset_embeddings[0].shape
    
    train_and_test(training_dataset, test_dataset, example_dim, retrain, save)                                                       
