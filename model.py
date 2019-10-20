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

print("Using TensorFlow version: {}\n".format(tf.__version__))

examples = None
labels = None

with open('datasets/dev_es.tsv') as tsvfile:
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    with open('temp_dataset.txt', 'w') as txt_dataset:
        txt_dataset.writelines([r['text'] for r in reader])
    tsvfile.seek(0)
    print("Computing FastText embeddings...\n")
    model = fasttext.train_unsupervised('temp_dataset.txt', model='skipgram')
    examples = np.empty((500,model.dim))
    labels_list = []
    first_row = True
    index = 0
    for row in reader:
        if first_row:
            first_row = False
            continue
        examples[index] = getTweetEmbedding(model, row['text'])
        labels_list.append(int(row['HS']))
        index += 1
    labels = np.asarray(labels_list)

print("\nDone!\n")

train_size = int(0.8 * examples.shape[0]) 

dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
dataset = dataset.shuffle(buffer_size=examples.shape[0])

train_data = dataset.take(train_size)
train_data = train_data.batch(64)

test_data = dataset.skip(train_size)
test_data = test_data.batch(64)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=examples[0].shape))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print("\nCompiling model...\n")

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model successfully compiled!\n")

model.fit(train_data, epochs=3)
model.evaluate(test_data)

""" examples_placeholder = tf.Variable(tf.zeros(examples.shape), name='examples', shape=examples.shape)
labels_placeholder = tf.Variable(tf.zeros(labels.shape), name='variables', shape=labels.shape) """

            
    


    