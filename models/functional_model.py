import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#### Pre-built models ####
def tass(inputs):
    x = layers.Dense(200, activation='relu')(inputs)

    return layers.Dropout(.2)(x)

def pereira_kohatsu(inputs):
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

##def lstm_pk(inputs):

def custom(inputs):
    d1 = layers.Dense(400, activation='relu')(inputs)

    lstm_inputs = tf.expand_dims(inputs, 1)
    lstm = layers.LSTM(400, stateful=True)(lstm_inputs)

    concat = layers.concatenate([d1, lstm], axis=1)

    return layers.Dense(400, activation='relu')(concat)

def FunctionalModel(inputShape):
    inputs = keras.Input(shape=inputShape)

    processed_inputs = conv_tass(inputs)

    final = pereira_kohatsu(processed_inputs)

    outputs = layers.Dense(1, activation='sigmoid')(final)

    model = keras.Model(inputs=inputs, outputs=outputs, name="main_output")

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

    return model