import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

DEFAULT_CHAR_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# prepare training data
def prepare_data(passwords, max_length=8, char_set=DEFAULT_CHAR_SET):
    # map each character to an integer
    char_to_int = {char: i for i, char in enumerate(char_set)}
    int_to_char = {i: char for i, char in enumerate(char_set)}

    X = []
    y = []
    for password in passwords:
        # convert to sequence of integers (one-hot encoding)
        encoded = [char_to_int[char] for char in password]
        X.append(encoded[:-1])  # All characters except last one (for input)
        y.append(encoded[1:])   # All characters except first one (for output)

    # Pad sequences to max_length
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_length, padding='pre')
    y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_length, padding='pre')
    return np.array(X), np.array(y), int_to_char, char_to_int

# Create the RNN model
def build_model(input_length, n_chars):
    model = Sequential()
    model.add(Embedding(input_dim=n_chars, output_dim=128, input_length=input_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(Dense(n_chars, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X, y, epochs=200, batch_size=64):
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

# Predict the next character
def predict_password(model, seed, max_length=8, char_to_int=None, int_to_char=None):
    result = seed
    for _ in range(max_length - len(seed)):
        # Convert the seed to an input vector
        encoded = [char_to_int[char] for char in result]
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=max_length, padding="pre")

        # Predict the next character
        predicted = model.predict(encoded, verbose=0)
        predicted_char_idx = np.argmax(predicted[0, -1, :])
        result += int_to_char[predicted_char_idx]

    return result

# Load the data
def load_data():
    # Dummy data for demonstration; replace this with actual password dataset
    passwords = ["password1", "123456", "qwerty", "abc123", "letmein"]
    return passwords

if __name__ == '__main__':
    # Example pasword dataset
    passwords = [
    'password', '12345', 'qwerty', 'letmein', 'welcome', 'admin', 'sunshine', 'iloveyou',
    '123123', 'qwerty123', 'password123', 'monkey', 'abc123', '1q2w3e4r', 'iloveyou123'
]
    
    max_length = 8
    X, y, int_to_char, char_to_int = prepare_data(passwords, max_length)

    model = build_model(max_length, len(char_to_int))
    train_model(model, X, y)

    # Predict a password based on seed
    seed = 'abc'
    predicted_password = predict_password(model, seed, max_length, char_to_int, int_to_char)
    
    print(f"Predicted password based on seed '{seed}': {predicted_password}")