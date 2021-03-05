import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0


def example2():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics='accuracy')

    model.fit(x_train, y_train, epochs=5)
    v_loss, v_accuracy = model.evaluate(x_test, y_test)

    print(f"[#]{v_accuracy}\n[#]{v_loss}")

    model.save("fashon_model.model")


def test_model(model):
    trained_model = tf.keras.models.load_model(model)
    pred = trained_model.predict([x_test])
    print(np.argmax(pred[20]))
    plt.imshow(x_test[20])
    plt.show()


test_model("./fashon_model.model")
