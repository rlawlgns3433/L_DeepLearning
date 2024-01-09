import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

file_path = 'image_data_with_labels.csv'
data = pd.read_csv(file_path)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(188,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

model.save('signal_classifier.h5')
