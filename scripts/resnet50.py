import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def list_files(startpath):
    for dirname, _, filenames in os.walk(startpath):
        print(f"Directory: {dirname}")
        # Show up to 5 filenames
        for filename in filenames[:5]:
            print(f"    {filename}")
        # Print the total number of files in the directory
        print(f"    Total files: {len(filenames)}\n")


async def build_model(image_size, batch_size, num_classes, learning_rate):
    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    train_generator = train_datagen.flow_from_directory(
        "./archive/colored_images",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )

    validation_generator = train_datagen.flow_from_directory(
        "./archive/colored_images",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )

    # Model Architecture
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    # Freeze base model layers
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Compile Model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        "best_resnet50_model.keras", monitor="val_accuracy", save_best_only=True
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Training
    await model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping],
    )

    # save
    model_path = "full_model.h5"  # Or any other desired filename
    weights_path = "weights.h5"  # Or any other desired filename
    model.save(model_path)
    model.save_weights(weights_path)

    return model


def load_resnet_model(model_path):
    model = load_model(model_path)
    print(get_summary(model))
    return model

def get_summary(model):
    return model.summary()