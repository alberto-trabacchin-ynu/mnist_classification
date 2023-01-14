import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import numpy as np

# Loading dataset: X_train = (60000, 28, 28), X_test = (10000, 28, 28)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape input data in 2D ndarr
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

# Define neural network model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(28*28,), activation='softmax')])

# Define compiler with loss function and metrics
model.compile(optimizer='Adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model defining epochs and validate it
model.fit( X_train_flattened, y_train,
           epochs=3,
           verbose=1,
           validation_data=(X_test_flattened, y_test) )

# Make a prediction with the trained model
predictions = model.predict(X_test_flattened)
print('Target: ' + str(y_test[0]))
print('Prediction: ' + str(np.argmax(predictions[0])))