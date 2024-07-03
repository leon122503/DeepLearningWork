from transformer import *
from load_text import load_prideandprejudice, load_warandpeace
from tokeniser import Tokeniser, plot_tok2vec
import numpy as np
import tensorflow as tf

from tokeniser import Tokeniser
from load_text import load_prideandprejudice
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout

from tensorflow.keras.callbacks import ModelCheckpoint
LoadFromFile =True  # Set to True to load the model from file, False to train the model

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


""" 
Rather than feeding just training data as in the previous assignments, for this assignment it's
best to use a custom written data generator.  This is a way for you to contol how the batches
of training data are created.  Here's a really simple data generator, that, in an epoch, randomly
picks words from text and creates a batch of training data of input and target output sequences of
fixed length.
"""


class predictTextDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, ids, seq_len, batch_size):
        """
        Constructor for the data generator.

        param ids: A list of integers representing the tokens in the training text.
        param seq_len: The length of the input and target sequences for the transformer model.
        param batch_size: The number of sequences in each batch for training
        """

        # Save all the training text and parameters of the data generator
        self.ids = ids
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Compute the number of samples - it's the length of the text minus the sequence length
        self.num_samples = len(self.ids) - seq_len - 1
        # Run the on_epoch_end() method - which scrambles the data into the batchs
        # (this method will also be run during trainin at the end of each training epoch)
        self.on_epoch_end()

    def __len__(self):
        """
        You must provide this method to tell the model how many batches there are in an epoch.

        returns The number of batches in an epoch.
        """
        return self.num_samples // self.batch_size

    def __data_generation(self, list_IDs_temp):
        """
        This method generates a batch of training data for the model.  It's called by the
        __getitem__() method which is called by the model during training.

        param list_IDs_temp: A list of integers representing the indexes of the training data
        to be included in the batch.
        returns A tuple of input and target output sequences for the model.
        """

        # The input and target sequences are both of shape (batch_size, seq_len) and
        # are integer ids of the tokens (the transformer model will convert these to word vectors based
        # on the embedding you specify)
        X = np.zeros((self.batch_size, self.seq_len), dtype="int")
        y = np.zeros((self.batch_size, self.seq_len), dtype="int")

        # For each index in the list of indexes...
        for i, ID in enumerate(list_IDs_temp):
            # ...get the sequence of tokens from the training of length seq_len starting at
            # index ID.  In this case the input sequence is the sequence spans the entire
            # length of seq_len, but you might also train on shorter sequences, padded with zeros.
            # makse_loss will included padded inputs/outputs.
            X[i, :seq_len] = self.ids[ID : ID + seq_len]
            # ....and the sequence of target tokens, which is the sequence of tokens from the
            # training text of length seq_len starting at index ID+1 (offset by one, to match
            # the next word in the output to current word in the input)
            y[i, :seq_len] = self.ids[ID + 1 : ID + seq_len + 1]

        return X, y

    def __getitem__(self, index):
        """
        This method is called by the model during training to get a batch of training data.

        param index: The index of the batch to get.
        returns A tuple of input and target output sequences for the model.
        """

        # Generate indexes of the batch
        list_IDs_temp = self.list_IDs[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        This method is called at the end of each epoch of training.  It shuffles the data
        so that the batches are different in each epoch.
        """

        # Shuffle the tokens
        self.list_IDs = np.arange(self.num_samples)
        np.random.shuffle(self.list_IDs)


#load text
text = load_prideandprejudice()
# Set variables
vocab_size = 600
seq_len = 15
vec_dim = 200
epochs = 75
# Load the tokeniser
tokeniser = Tokeniser.load("vocab600.json")
# Encode the text
ids = tokeniser.encode(text)
# Load the word2vec model
word2vec = load_model("600,200skipgram100epcoh.h5")
# Create a data generator
print("Loading data generator...")
train_data = predictTextDataGenerator(ids=ids, seq_len=seq_len, batch_size=32)

# Get the vocabulary size of the tokeniser
vocab_size = tokeniser.vocab_size

# Fetch the embedding matrix for the BERT tokeniser
w = word2vec.layers[0].get_weights()[0]
# Create a new sequential model
model = tf.keras.models.Sequential()



# Add the embedding layer to the model
model.add(FixedEmbedding(w, seq_len))

# Add positional encoding to the embedding
model.add(PositionalEncoding(vec_dim=vec_dim, seq_len=seq_len))

# Add the transformer layer to the model
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=6, dff=256))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=6, dff=256))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=6, dff=256))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=6, dff=256))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=6, dff=256))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=6, dff=256))

# Add the final dense layer to the model
model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))

# Custom learning rate schedule
learning_rate = CustomSchedule(vec_dim)
opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Compile the model
model.compile(optimizer=opt, loss=masked_loss, metrics=[masked_accuracy])

# Show the architecture of the model
model.summary()

if LoadFromFile:
    # Load the model
    model.load_weights("FixedEmbeddingsTransformer.h5")
else:
    # Train the model
    print("Training model...")
    model.fit(train_data, epochs=epochs)
    model.save_weights("FixedEmbeddingsTransformer.h5")
 
# Test the model by generating text that follows this prompt
prompt = "Hi, what"

sys.stdout.flush()
print(prompt, end="")
# Encode prompt to tokens
tokens = tokeniser.encode(prompt)

# Generate text
for i in range(1, 300):
    if len(tokens) >= seq_len:
        tokens = tokens[-seq_len:]
    j = len(tokens) - 1
    if len(tokens) < seq_len:
        x = np.concatenate(
            [tokens, np.zeros((seq_len - len(tokens)), dtype="int")], axis=0
        )
    else:
        x = np.array(tokens)
    x = np.expand_dims(x, axis=0)
    y = model.predict(x, verbose=False)
    y = np.argmax(y[:, j, :])
    t = tokeniser.decode(y)
    print(t, end="")
    sys.stdout.flush()
    tokens.append(y)

print("\n")
