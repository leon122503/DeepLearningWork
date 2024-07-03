import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from sklearn.metrics import confusion_matrix
import pickle, gzip
import load_oxford_flowers102
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
)

# load data
load_from_file = True
train_data, validation_data, test_data, class_names = (
    load_oxford_flowers102.load_oxford_flowers102(imsize=64, fine=False)
)
x_train = train_data["images"]
y_train = train_data["labels"]
x_test = validation_data["images"]
y_test = validation_data["labels"]

train_file_names = train_data["file_names"]
test_file_names = validation_data["file_names"]
# normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0
from tensorflow.keras import layers

if load_from_file:
    # Load the model
    model = tf.keras.models.load_model("auto-enconder.h5")
else:
    # create the model
    def conv_block(inputs, num_filters):
        x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(num_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def encoder_block(inputs, num_filters):
        x = conv_block(inputs, num_filters)
        p = layers.MaxPooling2D()(x)
        return x, p

    def decoder_block(inputs, skip, num_filters):
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(
            inputs
        )
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, num_filters)
        return x

    # builds Unet Model
    def build_model():
        inputs = layers.Input(shape=(64, 64, 3))
        x1, p = encoder_block(inputs, 32)
        x2, p = encoder_block(p, 64)
        x3, p = encoder_block(p, 128)
        b1 = conv_block(p, 256)
        x = decoder_block(b1, x3, 128)
        x = decoder_block(x, x2, 64)
        x = decoder_block(x, x1, 32)
        outputs = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)
        return tf.keras.Model(inputs, outputs)

    # Build the model
    model = build_model()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    # Train the model
    model.fit(
        x_train,
        x_train,
        epochs=100,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test),
    )
    # Save the model
    model.save("auto-enconder.h5")
model.summary()
# Predict the test set
reconstructed_images = model.predict(x_test)
# Calculate the error
error = np.square(np.subtract(x_test, reconstructed_images))

# Calculate the mean and standard deviation of the error
mean_error = np.mean(error)
std_error = np.std(error)
print(f"Mean error: {mean_error}")
print(f"Standard deviation of error: {std_error}")
n = 8  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Original")
    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_images[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Reconstruction")
plt.show()
