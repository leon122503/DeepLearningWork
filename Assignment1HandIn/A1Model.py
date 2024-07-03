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
from sklearn.utils.class_weight import compute_class_weight

flowers102_class_names = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]



load_from_file = False
history_save_name = "ClassBalancedFine.hist"
train_data, validation_data, test_data, class_names = (
    load_oxford_flowers102.load_oxford_flowers102(imsize=96, fine=True)
)
x_train = train_data["images"]
y_train = train_data["labels"]
x_test = validation_data["images"]
y_test = validation_data["labels"]

train_file_names = train_data["file_names"]
test_file_names = validation_data["file_names"]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Count the number of items in each class in the training set
unique, counts = np.unique(y_train, return_counts=True)
train_counts = dict(zip(unique, counts))
print("Training set:", train_counts)

# Count the number of items in each class in the test set
unique, counts = np.unique(y_test, return_counts=True)
test_counts = dict(zip(unique, counts))
print("Test set:", test_counts)
plt.imshow(x_train[0])
plt.show()
# print(y_test)
# Calculate class weights
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)


# Convert class_weights to a dictionary to pass it to the fit method
class_weights = dict(enumerate(class_weights))
# Convert labels to one hot vectors
y_train_categorical = tf.keras.utils.to_categorical(y_train, 102)
y_test_categorical = tf.keras.utils.to_categorical(y_test, 102)
# Create an ImageDataGenerator instance with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
from keras.callbacks import ModelCheckpoint
# Create a callback that saves the best model's weights
checkpoint = ModelCheckpoint(
    filepath="ClassBalancedFine.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1,
)
# Fit the data generator on your training data
datagen.fit(x_train)

# Create a generator that will yield augmented batches of data
augmented_data_generator = datagen.flow(x_train, y_train_categorical, batch_size=32)
# Load the model from file if it exists
if load_from_file and os.path.exists("ClassBalancedFine.h5"):
    if os.path.isfile(history_save_name):
        with gzip.open(history_save_name) as f:
            history = pickle.load(f)
    else:
        history = []
# Otherwise, create a new model
else:
    model = tf.keras.models.Sequential()

    # Block 1
    model.add(
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(96, 96, 3))
    )
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # Block 3
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # Block 4
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(102, activation="softmax"))  # for binary classification

    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
# Train the model
    train_info = model.fit(
        augmented_data_generator,
        epochs=200,
        class_weight=class_weights,
        validation_data=(x_test, y_test_categorical),
        callbacks=[checkpoint],
    )
    # Save the history of the training
    history = train_info.history
    with gzip.open(history_save_name, "w") as f:
        pickle.dump(history, f)

# Evaluate the model
net = tf.keras.models.load_model("ClassBalancedFine.h5")
loss, accuracy = net.evaluate(x_test, y_test_categorical)
print("Loss: ", loss)
print("Accuracy: ", accuracy)
# Compute the confusion matrix
y_pred = net.predict(x_test)
# Convert predictions classes to one hot vectors
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert validation observations to one hot vectors
y_true = np.argmax(y_test_categorical, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
# import seaborn as sns
# plt.figure(figsize=(10, 7))
# sns.heatmap(confusion_mtx, annot=True, fmt="d")
# plt.title("Confusion matrix")
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

# Compute class accuracy
class_accuracy = np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
for i, acc in enumerate(class_accuracy):

    print(f"Accuracy for class {i+1}: {acc}")

acccc = 0
for i, acc in enumerate(class_accuracy):
    acccc += acc
# Compute average accuracy
print(f"Average accuracy: {acccc/102}")
# Plot training and validation accuracy over the course of training
if history != []:
    fh = plt.figure()
    ph = fh.add_subplot(1, 1, 1)
    ph.plot(history["accuracy"], label="accuracy")
    ph.plot(history["val_accuracy"], label="val_accuracy")
    ph.set_xlabel("Epoch")
    ph.set_ylabel("Accuracy")
    ph.set_ylim([0.5, 1])
    ph.legend(loc="lower right")
    plt.show()

# Compute output for 16 test images
predictedImages = net.predict(x_test[:16])
predictedImages = np.argmax(predictedImages, axis=1)

# Show true labels and predictions for 16 test images
show_methods.show_data_images(
    images=x_test[:16],
    labels=y_test_categorical[:16],
    predictions=predictedImages,
    class_names=flowers102_class_names,
)
plt.show()
