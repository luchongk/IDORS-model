import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#### Pre-built models ####
def tass(inputs):
    x = GlobalMaxPooling1D()(inputs)

    x = Dense(200, activation='relu')(x)
    x = Dropout(.2)(x)

    return x

def pereira_kohatsu(inputs):
    # The first LSTM layer is missing

    droppedInputs = layers.Dropout(.8)(inputs)

    d1 = layers.Dense(1600, activation='relu')(droppedInputs)

    d1 = layers.Dropout(.8)(d1)

    d2 = layers.Dense(100, activation='relu')(d1) 

    final = layers.Dropout(.4)(d2)

    return final


def custom(inputs):
    d1 = layers.Dense(400, activation='relu')(inputs)

    lstm_inputs = tf.expand_dims(inputs, 1)
    lstm = layers.LSTM(400)(lstm_inputs)

    concat = layers.concatenate([d1, lstm], axis=1)

    final = layers.Dense(400, activation='relu')(concat)

    return final

def FunctionalModel(inputShape):
    inputs = keras.Input(shape=inputShape)

    final = custom(inputs)

    outputs = layers.Dense(1, activation='sigmoid')(final)

    model = keras.Model(inputs=inputs, outputs=outputs, name="main_output")

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

    return model