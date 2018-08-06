import re
import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

def list_pictures(directory, ext='JPEG|jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

def load():
    with open("data/ILSVRC2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt", 'r') as f:
        labels = f.read().split('\n')
        labels = np.asarray(labels)

    imgs = []
    for picture in tqdm(list_pictures("data/ILSVRC2012/val/")):
        img = img_to_array(load_img(picture, target_size=(224, 224)))
        imgs.append(img)
    return imgs, labels
