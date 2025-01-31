from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(10, activation='relu', input_shape=(5,)),
    Dense(1, activation='sigmoid')
])

model.summary()