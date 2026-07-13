import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Training settings
LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 32

DATASET_PATH = "dataset"

# Data Augmentation
image_generator = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

# Training Dataset
train_data = image_generator.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

# Validation Dataset
validation_data = image_generator.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

print("Class Labels :", train_data.class_indices)

# Load MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

base_model.trainable = False

# Build Classifier
head = base_model.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(1, activation="sigmoid")(head)

model = Model(inputs=base_model.input, outputs=head)

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Started...\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# Save Model (Keras 3 format)
model.save("mask_detector.keras")

print("\nModel Saved Successfully!")

# Plot Accuracy
plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Face Mask Detection Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()