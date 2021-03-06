import os, tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#### Pre-built models ####
def tass(inputs):
    x = layers.Dense(200, activation='relu')(inputs)

    return layers.Dropout(.2)(x)

def haternet(inputs):
    droppedInputs = layers.Dropout(.8)(inputs)

    d1 = layers.Dense(1600, activation='relu')(droppedInputs)

    d1 = layers.Dropout(.8)(d1)

    d2 = layers.Dense(100, activation='relu')(d1) 

    return layers.Dropout(.4)(d2)

#### Sentence embedding generators ####
def conv_tass(inputs):
    branch1 = layers.Conv1D(200, 2, padding='valid', activation='relu', strides=1)(inputs)
    branch2 = layers.Conv1D(200, 3, padding='valid', activation='relu', strides=1)(inputs)
    branch3 = layers.Conv1D(200, 4, padding='valid', activation='relu', strides=1)(inputs)

    concat = layers.concatenate([branch1, branch2, branch3], axis=1)

    return layers.GlobalMaxPooling1D()(concat)

def lstm_haternet(inputs):
    lstm_layer = layers.LSTM(600)

    return lstm_layer(inputs)

def custom(inputs):
    d1 = layers.Dense(400, activation='relu')(inputs)

    lstm_inputs = tf.expand_dims(inputs, 1)
    lstm = layers.LSTM(400, stateful=True)(lstm_inputs)

    concat = layers.concatenate([d1, lstm], axis=1)

    return layers.Dense(400, activation='relu')(concat)

def FunctionalModel(inputShape, input2Shape, bertInputShape, use_bert):
    inputs = keras.Input(shape=inputShape, name='tweet_word_vectors')

    inputs2 = keras.Input(shape=input2Shape, name='tweet_vectors')

    input_array = [inputs, inputs2]

    norm = layers.BatchNormalization()(inputs)

    processed_inputs = lstm_haternet(norm)#conv_tass(norm)

    inputs2 = layers.BatchNormalization()(inputs2)

    tweet_vector_array = [processed_inputs, inputs2]
    
    if (use_bert):
        inputs3 = keras.Input(shape=bertInputShape, name='bert_vectors')

        input_array.append(inputs3)

        inputs3 = layers.BatchNormalization()(inputs3)

        tweet_vector_array.append(inputs3)

    tweet_vector = layers.concatenate(tweet_vector_array, axis=1)

    final = haternet(tweet_vector)#tass(tweet_vector)

    output = layers.Dense(1, activation="sigmoid")(final)

    model = keras.Model(inputs=input_array, outputs=output, name="main_output")

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.TruePositives(),
                    tf.keras.metrics.TrueNegatives(),
                    tf.keras.metrics.FalsePositives(),
                    tf.keras.metrics.FalseNegatives()
                ])

    return model