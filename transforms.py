import skimage as sk
from scipy import ndarray
from skimage import transform
from skimage import util
import random
import numpy as np
import pickle
import torch



def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 15% on the left and 15% on the right
    random_degree = random.uniform(-15, 15)
    return sk.transform.rotate(image_array, random_degree)
#

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)
#
# def horizontal_flip(image_array: ndarray):
#     # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
#     return image_array[:, ::-1]

def double_batch(train_data: ndarray, train_label: ndarray):

    num_transformations_to_apply = 1
    print("generating data....")
    for i in range(0, len(train_data), 6):
        img = train_data[i]
        img = random_noise(np.reshape(img, (int(np.sqrt(len(img))), -1)))
        label = train_label[i]
        img = np.reshape(img, (1, int(np.square(len(img)))))
        train_data = np.append(train_data, img, axis=0)
        train_label = np.append(train_label, [label], axis=0)
    for i in range(1, len(train_data), 6):
        img = train_data[i]
        img = random_rotation(np.reshape(img, (int(np.sqrt(len(img))), -1)))
        label = train_label[i]
        img = np.reshape(img, (1, int(np.square(len(img)))))
        train_data = np.append(train_data, img, axis=0)
        train_label = np.append(train_label, [label], axis=0)
    print('done.')
    pickle.dump(train_data, open('./obj/train_data.pkl', 'wb'))
    pickle.dump(train_label, open('./obj/train_label.pkl', 'wb'))
