import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Configuration 
DATASET_DIR = r"D:\ML Dataset"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "model.h5" 
LABELS_SAVE_PATH = "labels.json"

# Label Mapping:Mapping based on user provided list 
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

    # Data Generators with Augmentation
    # 80% Training, 20% Validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
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

    # Save Class Indices to Labels Mapping 
    class_indices = train_generator.class_indices
    idx_to_label = {}
    
    for folder_name, idx in class_indices.items():
        if folder_name in FOLDER_MAP:
            idx_to_label[idx] = FOLDER_MAP[folder_name]
        else:
            idx_to_label[idx] = folder_name # Fallback if folder not in map

    print(f"Detected {len(idx_to_label)} classes.")
    with open(LABELS_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=4)
    print(f"Saved labels to {LABELS_SAVE_PATH}")

    num_classes = len(idx_to_label)

    # Model Setup
    print("Building MobileNetV2 model...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Freeze base model initially
    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # MobileNetV2 needs inputs in the range [-1, 1].
    # Our generator was scaling images to [0, 1], which is not correct.
    # MobileNetV2’s preprocess_input takes 0–255 images and converts them to [-1, 1].
    # So we remove rescale=1/255 and use preprocess_input instead.
    # Re-create the generators using MobileNetV2 preprocessing.   
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x) # Optional extra dense layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Training 
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    print("Training finished.")
    # Optional: Fine-tuning
    # Unfreeze the base model
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10), # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Starting fine-tuning...")
    history_fine = model.fit(
        train_generator,
        epochs=10, # Additional epochs
        validation_data=validation_generator,
        callbacks=callbacks
    )
    print("Fine-tuning finished.")

if __name__ == "__main__":
    main()