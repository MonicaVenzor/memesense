import os
import cv2
import numpy as np
import pandas as pd

from memesense.params import *

# Preprocess images

def preprocess_image(image, target_size=(224, 224)):
    img = cv2.imread(image)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0
    return img

def preprocess_images_and_labels(df, image_folder):
    images, labels = [], []
    for _, row in df.iterrows():
        image_path = os.path.join(image_folder, row['image_name'])
        img = preprocess_image(image_path)
        if img is not None:
            images.append(img)
            labels.append(row['overall_sentiment'])
    return np.array(images), labels
