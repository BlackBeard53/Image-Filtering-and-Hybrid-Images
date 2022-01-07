"""Unit tests for function create_1D_Gaussian_kernel in models.py"""

import torch
import numpy as np

from PIL import Image
from pathlib import Path
from scipy import ndimage
from proj1_code.student_code import my_median_filter

ROOT = Path(__file__).resolve().parent.parent  # ../..

def get_coin_img():
    coin_img_path = f'{ROOT}/data/coin.png'
    coin_img = Image.open(coin_img_path).convert('L')
    return coin_img

def test_median_filter():
    im_arr = np.arange(250).reshape(25, 10)
    im_torch = torch.tensor(im_arr, dtype=torch.float)

    # Actual result
    result = ndimage.median_filter(im_arr, size=(5, 5), mode='constant', cval=0)
    # Code output
    my_output = my_median_filter(im_torch, (5, 5))
    my_output = my_output.squeeze(2).numpy()
    
    assert np.linalg.norm(my_output - result) < 1e-5

def test_median_filter_with_variable_sizes():
    im_arr = np.arange(250).reshape(25, 10)
    im_torch = torch.tensor(im_arr, dtype=torch.float)

    # Actual result
    result = ndimage.median_filter(im_arr, size=(7, 5), mode='constant', cval=0)
    # Code output
    my_output = my_median_filter(im_torch, (7, 5))
    my_output = my_output.squeeze(2).numpy()
    
    assert np.linalg.norm(my_output - result) < 1e-5

def test_median_filter_with_size_one():
    im_arr = np.arange(250).reshape(25, 10)
    im_torch = torch.tensor(im_arr, dtype=torch.float)

    # Actual result
    result = ndimage.median_filter(im_arr, size=1, mode='constant', cval=0)
    # Code output
    my_output = my_median_filter(im_torch, 1)
    my_output = my_output.squeeze(2).numpy()
    
    assert np.linalg.norm(my_output - result) < 1e-5
