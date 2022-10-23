# https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

import tensorflow as tf
import keras
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

# Tensorboard
TrainingName = input("Enter the Tensorboard Model Name: ")
TrainingName = TrainingName.replace(" ", "_")
TrainingName = TrainingName if TrainingName != "" else time()
tensorboard = TensorBoard(log_dir="logs/{}".format(TrainingName))

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                rotation_range=40,
                                                                brightness_range=[0.2, 1.3],
                                                                shear_range=0.2,
                                                                fill_mode="nearest",)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    './MelanomalCancerDataset/train/',  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 300x300
    batch_size=128,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    './MelanomalCancerDataset/test/',  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 300x300
    batch_size=32,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(300, 300, 3), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])


# def lr_scheduler(epoch, lr):
#     if epoch < 18: # Might need to increase in early epochs, then decrease in later epochs
#         return lr * tf.math.exp(-0.003)
#     return lr * tf.math.exp(-0.03) # Refer to iteration 20 to improve the learning rate
def lr_scheduler(epoch, lr):
    if epoch < 4:
        return lr * tf.math.exp(0.03)
    if epoch < 20:  # Might need to increase in early epochs, then decrease in later epochs
        return lr * tf.math.exp(-0.003)
    return lr * tf.math.exp(-0.03)  # Refer to iteration 20 to improve the learning rate


lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
model.compile("adam", "binary_crossentropy", metrics=['accuracy', 'mae'])

model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=300,
    validation_data=validation_generator,
    validation_steps=8,
    verbose=1,
    callbacks=[tensorboard, lr_schedule]
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
