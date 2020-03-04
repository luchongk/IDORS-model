from __future__ import absolute_import, division, print_function, unicode_literals

import re, sys
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

class TfModel(Model):
  def __init__(self, example_dim):
    super(TfModel, self).__init__()
    self.flatten = Flatten(input_shape=example_dim)
    self.d1 = Dense(16, activation='relu')
    self.d2 = Dense(1, activation='sigmoid')

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

@tf.function
def train_step(model, tweets, labels, loss_object, optimizer, train_loss, train_accuracy, train_precision, train_recall):
    with tf.GradientTape() as tape:
        predictions = model(tweets, training=True)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    train_precision(labels, predictions)
    train_recall(labels, predictions)

@tf.function
def test_step(model, tweets, labels, loss_object, test_loss, test_accuracy, test_precision, test_recall):
    predictions = model(tweets)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_precision(labels, predictions)
    test_recall(labels, predictions)
                                          
