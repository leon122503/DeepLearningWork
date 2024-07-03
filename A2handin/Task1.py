"""
This script trains a skip-gram word embedding model using TensorFlow and Keras.
It loads a text file, tokenizes it, and generates pairs of target and context words.
The target and context words are then converted to one-hot encoding.
A neural network model is created and trained using the one-hot encoded data.
The trained model is saved and the word vectors are visualized using a plot.
"""

from load_text import load_prideandprejudice, load_warandpeace
from tokeniser import Tokeniser, plot_tok2vec

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from keras.models import load_model
from keras.utils import to_categorical

# Loading from file
load_tokeniser = True
load_mode1 = False

# Read in text
text = load_prideandprejudice()

# Set fixed vocab size
vocab_size = 600

# If loading
if load_tokeniser:
    # Load the tokeniser
    tokeniser = Tokeniser.load("vocab600.json")
else:
    # Create a tokeniser
    tokeniser = Tokeniser(vocab_size)
    tokeniser.train(text, verbose=True)
    tokeniser.save("vocab600.json")

ids = tokeniser.encode(text, verbose=True)
window_size = 2
pairs = []

# Generate pairs of target and context words in skip-gram fashion
for i in range(0, len(ids) - 1):
    for j in range(-window_size, window_size + 1):
        if j != 0:
            if i + j >= 0 and i + j < len(ids):
                pairs.append([ids[i], ids[i + j]])
pairs = np.array(pairs)

# Separate target and context words
target, context = zip(*pairs)

# Convert target and context to one-hot encoding
target_one_hot = to_categorical(target, num_classes=vocab_size)
context_one_hot = to_categorical(context, num_classes=vocab_size)


if load_mode1:
    # Load the model
    model = load_model("600,200skipgram100epcoh.h5")
else:
    # Create a neural network model
    model = Sequential()
    model.add(
        Dense(200, input_shape=(vocab_size,), activation="linear", use_bias=False)
    )
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()
    # Train the model
    history = model.fit(target_one_hot, context_one_hot, epochs=100, batch_size=64)
    model.save("600,200skipgram100epcoh.h5")

model.summary()
# Get the word vectors
w = model.layers[0].get_weights()[0]
print(w.shape)

# Visualize word vectors
plot_tok2vec(w, tokeniser.word_index, num_words_to_show=100)

# Plot the loss per epoch
import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
