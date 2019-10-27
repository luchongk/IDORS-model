from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import fasttext
import numpy as np
import re

def get_dataset():
    parsing_regex = re.compile(r'^__label__(\d)\s{1}(.*)$')

    train_file = open('training_set.txt')
    test_file = open('test_set.txt')

    _training_dataset = []
    _testing_dataset = []

    for line in train_file:
        match = re.match(parsing_regex, line)
        _training_dataset.append((match[2], match[1]))

    for line in test_file:
        match = re.match(parsing_regex, line)
        _testing_dataset.append((match[2], match[1]))

    return _training_dataset, _testing_dataset

def dataset_to_embeddings(_dataset, _ft_model):
    result = []
    for example in _dataset:
        result.append((get_tweet_embedding(example[0], _ft_model), example[1]))

    return result
    
def get_tweet_embedding(tweet, _ft_model):
    vec = np.repeat(0.0, _ft_model.dim)
    words = tweet.split()
    for word in words:
        vec += np.array(_ft_model[word])
    return np.divide(vec, len(words))

if __name__ == "__main__":
    training_dataset, test_dataset = get_dataset()
    ft_model = fasttext.load_model('baseline.bin')

    first_tweet_split = training_dataset[0][0].split()
    first_tweet_split_embeddings = []
    for word in first_tweet_split:
        first_tweet_split_embeddings.append(ft_model[word])
    
    print(len(first_tweet_split_embeddings), len(first_tweet_split))