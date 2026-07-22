import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------------------------------
# Dataset Path
# -------------------------------
dataset_path = r"dataset/sample_data"

# -------------------------------
# Parameters
# -------------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# -------------------------------
# Data Generator
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# -------------------------------
# CNN Model
# -------------------------------
model = Sequential([
    tf.keras.Input(shape=(64,64,3)),

    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(64, activation='relu'),

    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

# -------------------------------
# Compile Model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Model Summary
# -------------------------------
model.summary()

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# -------------------------------
# Save Model
# -------------------------------
os.makedirs("models", exist_ok=True)

model.save("models/trafficsense.keras")

# -------------------------------
# Output Folder
# -------------------------------
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# Accuracy Graph
# -------------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'],
         marker='o',
         label='Training Accuracy')

plt.plot(history.history['val_accuracy'],
         marker='o',
         label='Validation Accuracy')

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("outputs/accuracy.png")

plt.show()

# -------------------------------
# Loss Graph
# -------------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history['loss'],
         marker='o',
         label='Training Loss')

plt.plot(history.history['val_loss'],
         marker='o',
         label='Validation Loss')

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("outputs/loss.png")

plt.show()

print("\n===================================")
print("Training Completed Successfully!")
print("===================================")

print("Model Saved : models/trafficsense.keras")
print("Accuracy Graph : outputs/accuracy.png")
print("Loss Graph : outputs/loss.png")