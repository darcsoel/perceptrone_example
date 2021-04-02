import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(3, 3)),
        tf.keras.layers.Dense(9, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ])

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
    y = np.array([0, 0, 1])

    model.compile()
    model.fit(x, y, epochs=5)

    p = model.predict([[1, 1, 1]])
    print(p)
