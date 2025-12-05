import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Configuration
DATASET_DIR = r"C:\Users\ashra\Desktop\resnet_depi\dataset"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "resnet_model.h5"
LABELS_SAVE_PATH = "labels.json"

# Folder → Label mapping
FOLDER_MAP = {
    "0032": "alif", "0033": "baa", "0034": "ta", "0035": "tha",
    "0036": "jiim", "0037": "haa", "0038": "kha", "0039": "daal",
    "0040": "thal", "0041": "raa", "0042": "zay", "0043": "siin",
    "0044": "shiin", "0045": "saad", "0046": "daad", "0047": "taa",
    "0048": "zaa", "0049": "ayn", "0050": "ghayn", "0051": "faa",
    "0052": "qaaf", "0053": "kaaf", "0054": "laam", "0055": "miim",
    "0056": "noon", "0057": "haa", "0058": "waaw", "0059": "yaa"
}

def main():
    print(f"Checking dataset directory: {DATASET_DIR}")
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return

    # Data Generators (with augmentation)
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    print("Loading validation data...")
    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # class → label mapping
    class_indices = train_generator.class_indices
    idx_to_label = {}

    for folder_name, idx in class_indices.items():
        if folder_name in FOLDER_MAP:
            idx_to_label[idx] = FOLDER_MAP[folder_name]
        else:
            idx_to_label[idx] = folder_name

    print(f"Detected {len(idx_to_label)} classes.")

    with open(LABELS_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=4)

    print(f"Saved labels to {LABELS_SAVE_PATH}")

    num_classes = len(idx_to_label)

    # BUILD RESNET MODEL 
    print("Building ResNet50 model...")

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = False  # freeze initially

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # training callbacks
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    print("Starting training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    print("Training finished.")

    # FINE-TUNING 
    print("Starting fine-tuning...")

    base_model.trainable = True
    fine_tune_at = 100  # unfreeze last layers

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    print("Fine-tuning finished.")
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
