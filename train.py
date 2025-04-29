import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, applications, callbacks, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# Configuration
CONFIG = {
    "dataset_path": "chest_xray",
    "image_size": (224, 224),
    "batch_size": 32,
    "num_classes": 4,
    "classes": ["COVID-19", "Pneumonia", "Tuberculosis", "Normal"],
    "augmentation": {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "horizontal_flip": True
    },
    "model": {
        "base_model": "ResNet50",
        "dropout_rate": 0.5,
        "dense_units": 512,
        "learning_rate": 1e-4,
        "fine_tune_at": 100
    },
    "resume_from_epoch": 25  # Set manually if resuming
}

def create_data_pipeline():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        **CONFIG["augmentation"]
    )

    train_generator = train_datagen.flow_from_directory(
        CONFIG["dataset_path"],
        target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        class_mode='categorical',
        subset='training',
        color_mode='rgb'
    )

    val_generator = train_datagen.flow_from_directory(
        CONFIG["dataset_path"],
        target_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        class_mode='categorical',
        subset='validation',
        color_mode='rgb'
    )

    return train_generator, val_generator

def build_model():
    base_model = getattr(applications, CONFIG["model"]["base_model"])(
        include_top=False,
        weights='imagenet',
        input_shape=(*CONFIG["image_size"], 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*CONFIG["image_size"], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(CONFIG["model"]["dense_units"], activation='relu')(x)
    x = layers.Dropout(CONFIG["model"]["dropout_rate"])(x)
    outputs = layers.Dense(CONFIG["num_classes"], activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    base_model.trainable = True
    for layer in base_model.layers[:CONFIG["model"]["fine_tune_at"]]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["model"]["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(name='auc')
        ]
    )
    return model

def train_model():
    train_gen, val_gen = create_data_pipeline()
    model = build_model()

    cb = [
        callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_auc', mode='max'),
        callbacks.ModelCheckpoint("latest_checkpoint.h5", save_best_only=False),
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
        callbacks.TensorBoard(log_dir='logs'),
        callbacks.CSVLogger('training_log.csv', append=True)
    ]

    class_counts = train_gen.classes
    class_weights = {
        i: len(class_counts) / (CONFIG["num_classes"] * count)
        for i, count in enumerate(np.bincount(class_counts))
    }

    resume_path = "latest_checkpoint.h5"
    initial_epoch = 0

    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        model.load_weights(resume_path)
        initial_epoch = CONFIG["resume_from_epoch"]

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // CONFIG["batch_size"],
        validation_data=val_gen,
        validation_steps=val_gen.samples // CONFIG["batch_size"],
        epochs=50,
        initial_epoch=initial_epoch,
        callbacks=cb,
        class_weight=class_weights
    )

    return model, history

def create_gradio_interface(model):
    def predict(image):
        img = tf.image.resize(image, CONFIG["image_size"])
        img = tf.expand_dims(img / 255.0, axis=0)
        preds = model.predict(img)[0]
        return {CONFIG["classes"][i]: float(preds[i]) for i in range(CONFIG["num_classes"])}

    example_images = [os.path.join("examples", f) for f in os.listdir("examples")] if os.path.exists("examples") else None

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(image_mode='RGB', height=CONFIG["image_size"][0], width=CONFIG["image_size"][1]),  # Use 'shape', not 'image_size'
        outputs=gr.Label(num_top_classes=CONFIG["num_classes"]),
        title="Chest X-ray Disease Classifier",
        description="Classify chest X-rays into COVID-19, Pneumonia, Tuberculosis, or Normal",
        examples=example_images
    )

if __name__ == "__main__":
    model, history = train_model()
    model.save("final_model.keras")  # Save in Keras format
    interface = create_gradio_interface(model)
    interface.launch(share=True)