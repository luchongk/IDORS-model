import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def FunctionalModel(inputShape):
    inputs = keras.Input(shape=inputShape)
    d1 = layers.Dense(16, activation='relu')(inputs)
    outputs = layers.Dense(1, activation='sigmoid')(d1)

    model = keras.Model(inputs=inputs, outputs=outputs, name="test")

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

    return model