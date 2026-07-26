from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

# ======================================================
# Configuration
# ======================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

DATASET_DIR = "datasets"

PART1 = os.path.join(DATASET_DIR, "HAM10000_images_part_1")
PART2 = os.path.join(DATASET_DIR, "HAM10000_images_part_2")
CSV_PATH = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")

# ======================================================
# Load Metadata
# ======================================================

print("=" * 60)
print("Loading HAM10000 Metadata...")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

print(df.head())

print("\nTotal Images:", len(df))

print("\nDisease Distribution:")
print(df["dx"].value_counts())


# ======================================================
# Find Image Paths
# ======================================================

def get_image_path(image_id):
    image_name = image_id + ".jpg"

    path1 = os.path.join(PART1, image_name)
    path2 = os.path.join(PART2, image_name)

    if os.path.exists(path1):
        return path1

    if os.path.exists(path2):
        return path2

    return None


df["image_path"] = df["image_id"].apply(get_image_path)

missing = df["image_path"].isnull().sum()

print("\nMissing Images:", missing)

if missing > 0:
    print(df[df["image_path"].isnull()].head())
    raise ValueError("Some images are missing.")

print("All images found successfully.")

# ======================================================
# Encode Labels
# ======================================================

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["dx"])

print("\nClasses:")
print(list(encoder.classes_))

# ======================================================
# Train Validation Split
# ======================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining Images:", len(train_df))
print("Validation Images:", len(val_df))

# ======================================================
# One Hot Encoding
# ======================================================

train_labels = to_categorical(train_df["label"])
val_labels = to_categorical(val_df["label"])

NUM_CLASSES = train_labels.shape[1]

print("\nNumber of Classes:", NUM_CLASSES)

# ======================================================
# TensorFlow Dataset
# ======================================================

print("\nCreating TensorFlow Dataset...")


def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    image = preprocess_input(image)

    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_df["image_path"].values,
        train_labels
    )
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_df["image_path"].values,
        val_labels
    )
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (
    train_dataset
    .shuffle(1000)
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("TensorFlow Dataset Created Successfully.")

# ======================================================
# Test Dataset
# ======================================================

print("\nTesting Dataset Pipeline...\n")

for images, labels in train_dataset.take(1):
    print("Image Batch Shape :", images.shape)
    print("Label Batch Shape :", labels.shape)

print("\nDataset Pipeline Working Successfully!")
print("=" * 60)

print("\nBuilding MobileNetV2 Model...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("app/models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "app/models/skin_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nStarting Training...\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(val_dataset)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

model.save("app/models/skin_model_final.keras")

print("\nTraining Completed!")
print("Best model saved as:")
print("app/models/skin_model.keras")
