# https://www.kaggle.com/datasets/fusicfenta/chest-xray-for-covid19-detection

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

TrainingName = input("Enter the Tensorboard Model Name: ")
TrainingName = TrainingName.replace(" ", "_")
TrainingName = TrainingName if TrainingName != "" else time()
tensorboard = TensorBoard(log_dir="logs/{}".format(TrainingName))


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255,
                                                                rotation_range=5,
                                                                brightness_range=[1, 1.3],
                                                                shear_range=0.2,
                                                                fill_mode="nearest")

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory("Covid Xray Dataset/Train/",
                                                    target_size=(300, 300),
                                                    batch_size=32,
                                                    shuffle=True,
                                                    class_mode="binary",)

validation_generator = validation_datagen.flow_from_directory("Covid Xray Dataset/Val/",
                                                         target_size=(300, 300),
                                                         batch_size=32,
                                                         shuffle=True,
                                                         class_mode="binary")


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(300, 300, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])


model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), "binary_crossentropy", metrics=['accuracy', 'mae'])

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
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

