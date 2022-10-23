# https://www.kaggle.com/datasets/vardhilpatel/skin-cancer-isic-dataset-2018?select=Fold_1

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

TrainingName = input("Enter the Tensorboard Model Name: ")
TrainingName = TrainingName.replace(" ", "_")
TrainingName = TrainingName if TrainingName != "" else time()
tensorboard = TensorBoard(log_dir="logs/{}".format(TrainingName))

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255,
                                                                rotation_range=40,
                                                                brightness_range=[0.7, 1.3],
                                                                zoom_range=[0.5, 1.3],
                                                                shear_range=0.2,
                                                                fill_mode="nearest")

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory("SkinCancerDataset/Training/Lesions/",
                                                    target_size=(200, 200),
                                                    batch_size=32,
                                                    shuffle=True,
                                                    class_mode="categorical",)

validation_generator = validation_datagen.flow_from_directory("SkinCancerDataset/Validation/Lesions",
                                                         target_size=(200, 200),
                                                         batch_size=32,
                                                         shuffle=True,
                                                         class_mode="categorical")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(200, 200, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=7, activation='softmax')
])

model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), "categorical_crossentropy", metrics=['accuracy', 'mae'])

model.fit(
    train_generator,
    steps_per_epoch=64,
    validation_data=validation_generator,
    validation_steps=16,
    epochs=1400,
    verbose=1,
    callbacks=[tensorboard]
)

# Ask to save the model
while True:
    save_model = input("Save the model? (y/n): ")
    if save_model == "y":
        name = input("Name of the model: ")
        model.save(name + ".h5")
        print("Model saved!")
        exit(0)
    elif save_model == "n":
        print("Model not saved!")
        exit(0)
    else:
        print("Invalid input! Try again!")
