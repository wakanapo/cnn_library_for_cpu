import re
import os
import scipy.io
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from tqdm import tqdm

def load():
    matdata = scipy.io.loadmat("data/ILSVRC2012/ILSVRC2012_devkit_t12/data/meta.mat")
    with open("data/ILSVRC2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt", 'r') as f:
        labels = [matdata['synsets'][int(x)-1]['WNID'][0][0] for x in f]

    imgs = []
    for i in tqdm(range(len(labels))):
        picture_name = "data/ILSVRC2012/val/ILSVRC2012_val_000{0:05d}.JPEG".format(i+1)
        img = img_to_array(load_img(picture_name, target_size=(224, 224)))
        imgs.append(img)
    imgs = np.asarray(imgs)
    return imgs, labels
