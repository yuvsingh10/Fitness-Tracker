import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import random

# ---- Config ----
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
FREEZE_LAYERS_UPTO = 100  # fine-tune upper layers

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / 'data' / 'splits'
TRAIN_TXT = SPLIT_DIR / 'train.txt'
VAL_TXT = SPLIT_DIR / 'val.txt'
CLASS_JSON = SPLIT_DIR / 'class_indices.json'
MODEL_OUT = BASE_DIR / 'models' / 'mobilenetv2_correctness.h5'


# ---- Data Loader from txt (frame_path, class_id) ----
def parse_list(txt_path):
    items = []
    with open(txt_path, 'r') as f:
        for line in f:
            fp, cid = line.strip().rsplit(' ', 1)
            items.append((fp, int(cid)))
    return items


train_items = parse_list(TRAIN_TXT)
val_items = parse_list(VAL_TXT)

random.shuffle(train_items)

num_classes = max([cid for _, cid in train_items + val_items]) + 1


# ---- TF Dataset ----
def decode_img(fp, label):
    img = tf.io.read_file(fp)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    return img, label


# Data augmentation
def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_crop(tf.image.resize_with_pad(img, 240, 240),
                               size=(224, 224, 3))
    return img, label


def make_ds(items, training=True):
    paths = [fp for fp, _ in items]
    labels = [cid for _, cid in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(items), 5000))
    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds(train_items, training=True)
val_ds = make_ds(val_items, training=False)


# ---- Model ----
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True

# Freeze some layers
for layer in base_model.layers[:FREEZE_LAYERS_UPTO]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ---- Training ----
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_OUT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("✅ Training complete. Best model saved to", MODEL_OUT)
