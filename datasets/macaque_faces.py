# -*- coding: utf-8 -*-


import numpy as np


def load_data(data_type='mini'):
    """Loads the macaque faces dataset.

    # Arguments
        data_type: mini or full

    # Returns
        Numpy arrays, with shape (samples, h, w, channels).
    """
    
    data_path = 'D:/DataSets/macaque_faces/mf3/'
    if data_type == 'full':
        data_file = 'mf3.npy'
    else:
        data_file = 'mf3_mini.npy'
        
    data = np.load(data_path+data_file)

    return data
