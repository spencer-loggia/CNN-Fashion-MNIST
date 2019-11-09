""" Simple data loader :) """

import numpy as np

def load(path, split, load_labels=True):
    if load_labels:
        labels = np.load(f'{path}/{split}.labels.npy')
    else:
        labels = None
    data = np.load(f'{path}/{split}.feats.npy')
    return data, labels
