from __future__ import absolute_import, division, print_function, unicode_literals

import re, sys
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

class TfModel(Model):
    def __init__(self, 
                example_dim, 
                loss_object, 
                optimizer, 
                train_loss, 
                train_accuracy, 
                train_precision, 
                train_recall, 
                test_loss, 
                test_accuracy, 
                test_precision, 
                test_recall):
        super(TfModel, self).__init__()

        self.flatten = Flatten(input_shape=example_dim)
        self.d1 = Dense(16, activation='relu')
        self.d2 = Dense(1, activation='sigmoid')

        self.loss_object = loss_object 
        self.optimizer = optimizer
            
        self.train_loss = train_loss 
        self.train_accuracy = train_accuracy 
        self.train_precision = train_precision 
        self.train_recall = train_recall

        self.test_loss = test_loss 
        self.test_accuracy = test_accuracy 
        self.test_precision = test_precision 
        self.test_recall = test_recall

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    @tf.function
    def train_step(self, tweets, labels):
        with tf.GradientTape() as tape:
            predictions = self.call(tweets)
            loss = self.loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        self.train_precision(labels, predictions)
        self.train_recall(labels, predictions)

    @tf.function
    def test_step(self, tweets, labels):
        predictions = self(tweets)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        self.test_precision(labels, predictions)
        self.test_recall(labels, predictions)

    def fit(self, training_ds, test_ds, epochs):
        for epoch in range(epochs):
            for tweets, labels in training_ds:
                self.train_step(tweets, tf.reshape(labels, [64,1]))  

            for test_tweets, test_labels in test_ds:
                self.test_step(test_tweets, tf.reshape(test_labels, [64,1]))

            template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Train Precision: {}, Train Recall: {}, Train F-score: {}, Test Loss: {}, Test Accuracy: {}, Test Precision: {}, Test Recall: {}, Test F-Score: {}'
            
            tr_precision = self.train_precision.result()*100
            tr_recall = self.train_recall.result()*100
            train_fscore = 2 * (tr_precision * tr_recall / (tr_precision + tr_recall))

            tst_precision = self.test_precision.result()*100
            tst_recall = self.test_recall.result()*100
            test_fscore = 2 * (tst_precision * tst_recall / (tst_precision + tst_recall))
            
            print(template.format(epoch+1,
                                self.train_loss.result(),
                                self.train_accuracy.result()*100,
                                tr_precision,
                                tr_recall,
                                train_fscore,
                                self.test_loss.result(),
                                self.test_accuracy.result()*100,
                                tst_precision,
                                tst_recall,
                                test_fscore))

            self.train_loss.reset_states() 
            self.train_accuracy.reset_states()
            self.train_precision.reset_states()
            self.train_recall.reset_states()

            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.test_precision.reset_states()
            self.test_recall.reset_states()   
                                          
