# Extract selected folder from HAM10000 dataset into image subcategories

import os
import shutil
import sys
import time

import PIL.Image
import numpy as np
import tensorflow as tf

import pandas as pd
import math

# Define the path to the HAM10000 dataset
ROOT_PATH = r"C:\Users\Aiden\Desktop\Disease Recognition"
HAM_PATH = r"\dataverse_files"
SUB_PATH = r"\dataverse_files\HAM10000_images_part_"

EXTRACT_PATH = r"\dataverse_files\HAM10000_categories"

MAX_IMAGES = 10000  # Max Images Per Category

# Define the category names
categories = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
categoryAmounts = {"akiec": 0, "bcc": 0, "bkl": 0, "df": 0, "mel": 0, "nv": 0, "vasc": 0}


# Create a function to extract the images
def extract_images(category, rootPath, pullPath, extractPath, batches):
    # Read in the metadata and extract the image names
    # of the specified category
    df = pd.read_csv(rootPath + HAM_PATH + r"\HAM10000_metadata.csv")
    df = df[df.dx == category]
    images = df.image_id.values

    print("\n[INFO] extracting '{}' images...".format(category))
    i = 0
    # Copy the selected images to the extract path
    for image in range(len(images)):
        # Log the progress
        # check if image exists in the pull path
        if os.path.exists(ROOT_PATH + pullPath + "\\" + images[image] + ".jpg"):
            sys.stdout.write("\rExtracting image {} of category {}".format(i+1, category))
            i += 1
            shutil.copy(ROOT_PATH + pullPath + "\\" + images[image] + ".jpg", ROOT_PATH + extractPath + "\\" + category)
        if i >= round(MAX_IMAGES/batches):
            break
    categoryAmounts[category] += i
    pass


# Create the subdirectories
for category in categories:
    os.makedirs(ROOT_PATH + EXTRACT_PATH + "\\" + category, exist_ok=True)

# Extract the images
for category in categories:
    extract_images(category, ROOT_PATH, SUB_PATH + "1", EXTRACT_PATH, 2)
    extract_images(category, ROOT_PATH, SUB_PATH + "2", EXTRACT_PATH, 2)


print(categoryAmounts)

# Start image augmentation

dataGen = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.5, 1.2],
                                                            rotation_range=20,
                                                            zoom_range=0.2,
                                                            horizontal_flip=True,
                                                            vertical_flip=True,
                                                            fill_mode="nearest")


def augment_images(category, rootPath, extractPath):
    # Read in the metadata and extract the image names
    # of the specified category
    df = pd.read_csv(rootPath + HAM_PATH + r"\HAM10000_metadata.csv")
    df = df[df.dx == category]
    images = df.image_id.values

    print("\n[INFO] Augmenting '{}' images...".format(category))
    i = 0
    # Copy the selected images to the extract path
    amt = math.ceil(max(categoryAmounts.values())/categoryAmounts[category])
    if amt == 1:
        print("Process terminated for category {}".format(category))
        return

    for image in range(len(images)):
        # Log the progress
        if os.listdir(rootPath + extractPath + "\\" + category).__len__() >= MAX_IMAGES:
            print("\nProcess terminated for category {} - Image Overflow".format(category))
            break
        if os.path.exists(ROOT_PATH + extractPath + "\\" + category + "\\" + images[image] + ".jpg"):
            amtG = 0
            img = tf.keras.preprocessing.image.load_img(ROOT_PATH + extractPath + "\\" + category + "\\" + images[image] + ".jpg")
            input_arr = tf.keras.preprocessing.image.img_to_array(img)
            input_arr = np.array([input_arr])  # Convert single image to a batch

            for batch in dataGen.flow(input_arr,
                                      batch_size=amt,
                                      save_to_dir=ROOT_PATH + extractPath + "\\" + category,
                                      save_prefix="AUG_" + images[image], save_format="jpg"):
                amtG += 1
                if amtG >= amt or amtG >= MAX_IMAGES:
                    break

                sys.stdout.write("\rAugmenting image {} of category {}. Image Generated: {}".format(i+1, category, amtG))
            i += 1
            continue
        sys.stdout.write("\rAugmenting image {} of category {}. Image Generated: null".format(i+1, category))


# for i in categories:
    # augment_images(i, ROOT_PATH, EXTRACT_PATH)

# Count the number of images in each category
for category in categories:
    lenImages = len(os.listdir(ROOT_PATH + EXTRACT_PATH + "\\" + category))
    print("[INFO] {} images in '{}' category".format(lenImages, category))
    print("Class Weight: {}".format((1 / lenImages) * (sum(categoryAmounts.values()) / len(categories))))
