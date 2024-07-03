import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import show_methods
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
#load data
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
#normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0
from tensorflow.keras import layers

# Generate noisy images
def add_noise(img, noise_factor):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=img.shape)
    img_noisy = img + noise
    img_noisy = np.clip(img_noisy, 0.0, 1.0)
    return img_noisy



inpt_images = []
ouput = []
test_input = []
test_output = []
# Generate multiple versions of the first test image with various amounts of noise
for i in range(len(x_train)):
    image = [
        add_noise(x_train[i], noise_factor)
        for noise_factor in np.linspace(0, 1, num=20)
    ]
    # Generate pairs of consecutive images
    for j in range(len(image) - 1, 0, -1):
        inpt_images.append(image[j])
        ouput.append(image[j - 1])

# Generate multiple versions of the first test image with various amounts of noise
for i in range(len(x_test)):
    image = [
        add_noise(x_test[i], noise_factor) for noise_factor in np.linspace(0, 1, num=20)
    ]
    # Generate pairs of consecutive images
    for j in range(len(image) - 1, 0, -1):
        test_input.append(image[j])
        test_output.append(image[j - 1])
# Convert the lists to numpy arrays
input_images = np.array(inpt_images)
output = np.array(ouput)
test_input = np.array(test_input)
test_output = np.array(test_output)
print(input_images.shape)

# Define the model
if load_from_file:
    # Load the model from the file
    model = tf.keras.models.load_model("diffusion64x20.h5")
else:
    # Define the model
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
    #builds Unet Model
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
        input_images,
        output,
        epochs=100,
        batch_size=64,
        shuffle=True,
        validation_data=(test_input, test_output),
    )
    # Save the model
    model.save("diffusion.h5")
model.summary()

# Generate image from random noise
noise = np.random.rand(1,64, 64, 3)
generated_images = []
generated_images.append(noise)
# Generate 20 images by passing the image into the model 20 times
for i in range(20):
    noise = model.predict(noise)
    generated_images.append(noise)
#plot the images
for i in range(20):
    plt.subplot(4, 5, i+1)  
    plt.imshow(generated_images[i].reshape(64, 64, 3), cmap='gray')  
    plt.axis('off') 
plt.show()
# Generate image from normal distributed noise
noise = np.random.normal(loc=0.5, scale=0.5, size=(1,64, 64, 3))
generated_images = []
generated_images.append(noise)
# Generate 20 images by passing the image into the model 20 times
for i in range(20):
    noise = model.predict(noise)
    generated_images.append(noise)
#plot the images
for i in range(20):
    plt.subplot(4, 5, i+1)  
    plt.imshow(generated_images[i].reshape(64, 64, 3), cmap='gray')  
    plt.axis('off')  

plt.show()