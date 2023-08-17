"""
`databuilder.py` performs all of the necessary steps to properly load and transform the data into a form that our neural networks can use. I have extracted these steps out into a separate file so that we do not need to repeat these steps in our different network implementations and in order to simplify the coding of our networks.
"""

# Import the necessary libraries.
import torch
import cv2
import numpy as np
# Import our configurations from `config.py`.
import config as c

# Loads 3D data, transforming it as needed.
def prep_data(fp, dim):
    data = np.fromfile(fp, dtype = np.single)
    data = data.reshape(dim)
    output =np.flip(np.rot90(data, k = 1, axes = (1, 2)), axis = 1)
    return output

# Loads 4D data, transforming it as needed.
def transform_4d(arr, dim):
    arr = arr.reshape(dim)
    output_arr = np.empty(dim)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            phan = arr[i, j, :, :]
            new_phan = np.flip(np.flip(np.rot90(phan, k = -1), axis = 0), axis = 1)
            new_phan = np.transpose(new_phan)
            # Rotation 45 degrees using cv2
            center = (new_phan.shape[1] // 2, phan.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(45), 1.0)
            new_phan = cv2.warpAffine(new_phan, rotation_matrix, (new_phan.shape[1], new_phan.shape[0]))
            output_arr[i, j, :, :] = new_phan 
    return output_arr

# Load and transform the phantom data.
phantom_np_data = torch.load("data/masked_phan.bin")
phantom = torch.reshape(phantom_np_data, c.PHAN_DIM)

# Load and transform the unnormalized phantom data.
not_norm_data = torch.load("data/not_norm_masked_phan.bin")
not_norm = torch.reshape(not_norm_data, c.PHAN_DIM)

# Load and transform the system matrix.
lor_np_data = np.fromfile("data/LoR9to1.bin", dtype = np.single)
lor_np = transform_4d(lor_np_data, c.LOR_DIM)
lor = torch.reshape((torch.from_numpy(lor_np)), c.LOR_DIM)
# Save the original, pretraining matrix for easy access for display.
og_lor = lor_np 

# Load and transform the ground truth data.
gt_np_data = np.fromfile("data/GTsim9to1.bin", dtype = np.single)
gt = torch.reshape((torch.from_numpy(gt_np_data)), c.GT_DIM)
gt = torch.t(gt)
