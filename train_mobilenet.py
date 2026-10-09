import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# =======================
# DATA AUGMENTATION
# =======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# =======================
# DATASET PATH (FIXED)
# =======================
DATASET_PATH = "C:/Users/ankit/Agro-AI/Agro-AI/PlantVillage"

# =======================
# TRAIN DATA
# =======================
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# =======================
# VALIDATION DATA
# =======================
val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

# =======================
# MODEL
# =======================
base_model = MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =======================
# TRAIN
# =======================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# =======================
# SAVE MODEL
# =======================
model.save("plant_disease_mobilenet.h5")

print("Training completed successfully!")