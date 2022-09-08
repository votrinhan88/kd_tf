import tensorflow as tf
keras = tf.keras
layers = keras.layers

def simple_conv() -> keras.Model:
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    # x = layers.Conv2D(128, (3, 3))(x)
    # x = layers.BatchNormalization()(x)
    # x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name = 'simple_conv')
    return model

if __name__ == '__main__':
    model = simple_conv()
    model.summary()