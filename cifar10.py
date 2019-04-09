# Cifar10Model : Résultats obtenus
# ==> loss: 0.9029 - acc: 0.7892

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

model = keras.Sequential([
    tf.layers.Conv2D(32, 5, 1, padding="same", use_bias=True, kernel_regularizer=keras.regularizers.l2(0.001),
                     activation=tf.nn.relu, input_shape=(32, 32, 3)),
    tf.layers.Conv2D(32, 5, 1, padding="same", use_bias=True, kernel_regularizer=keras.regularizers.l2(0.001),
                     activation=tf.nn.relu),
    tf.layers.BatchNormalization(),
    tf.layers.Conv2D(32, 5, 2, padding="same", use_bias=True, kernel_regularizer=keras.regularizers.l2(0.001),
                     activation=tf.nn.relu),
    tf.layers.Dropout(0.5),
    tf.layers.Conv2D(64, 4, 2, padding="same", use_bias=True, kernel_regularizer=keras.regularizers.l2(0.001),
                     activation=tf.nn.relu),
    tf.layers.BatchNormalization(),
    tf.layers.Conv2D(128, 4, 2, padding="same", use_bias=True, kernel_regularizer=keras.regularizers.l2(0.001),
                     activation=tf.nn.relu),
    tf.layers.Dropout(0.5),
    tf.layers.Flatten(),
    tf.layers.Dense(200, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
    tf.layers.BatchNormalization(),
    tf.layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.softmax)
])

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(datagen.flow(train_images, train_labels, batch_size=100),
                    epochs=30,
                    verbose=1,
                    validation_data=(test_images, test_labels)
                    )

tf.keras.models.save_model(
    model,
    filepath="cifar10Model.h5",
    overwrite=True,
    include_optimizer=True
)

history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, '--', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.show()