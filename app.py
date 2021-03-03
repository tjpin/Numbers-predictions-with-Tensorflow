import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def tf_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(x_train, y_train, epochs=3)
    validation_loss, validation_accuracy = model.evaluate(x_test, y_test)

    model.save('num_reader.model')  # save trained model

    print(f"[#]\n{validation_accuracy}\n[#]{validation_loss}")


def predictions(model):
    test_model = tf.keras.models.load_model(model)  # load pre_trained model
    preds = test_model.predict([x_test])
    print(np.argmax(preds[20]))
    plt.imshow(x_test[20])
    plt.show()


predictions("num_reader.model")
