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
        labels = np.asarray([int(x) for x in f])
    print("Labels: ", labels[:5])

    imgs = []
    for picture in tqdm(range(len(labels))):
        picture_name = "data/ILSVRC2012/val/ILSVRC2012_val_000{0:05d}.JPEG".format(i+1)
        img = img_to_array(load_img(picture_name, target_size=(224, 224)))
        imgs.append(img)
    imgs = np.asarray(imgs)
    return imgs, labels
