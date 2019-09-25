from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import csv
import fasttext
import numpy as np

def getTweetEmbedding(model, tweet):
    vec = np.repeat(1.0, 100)
    words = tweet.split()
    for word in words:
        vec += np.array(model[word])
    return np.divide(vec, len(words))

print(tf.__version__)
dataset = []
labels = []

with open('datasets/dev_es.tsv') as tsvfile:
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    tweets = [r['text'] for r in reader]
    labels = [r['HS'] for r in reader]
    with open('temp_dataset.txt', 'w') as txt_dataset:
        txt_dataset.writelines(tweets)
    model = fasttext.train_unsupervised('temp_dataset.txt', model='skipgram')
    print(model.dim)
    for tweet in tweets:
        dataset.append(getTweetEmbedding(model, tweet))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(len(dataset),100)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print(model.summary())

            
    


    