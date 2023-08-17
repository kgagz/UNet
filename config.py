"""
`config.py` holds all of the configuration variables for the networks implemented within this folder. These variables are changeable, but are "more permanent" than the ones found within `hyperparameters.py`. In other words, these variables are less likely to change between runs, so it makes sense them to extract them out from the other parameters that we manipulate more frequently.
"""

# Import necessary libraries.
import numpy as np
import torch

# Dimensions:
LENGTH = 75
HEIGHT = 75
SLICES = 25
PHANTOMS = 16
LORS = 4096
SIZE = (HEIGHT * LENGTH * SLICES)
DIM = (SLICES, HEIGHT, LENGTH)
PHAN_DIM = (PHANTOMS, SLICES, LENGTH, HEIGHT)
PHAN_RESHAPE = (PHANTOMS, SIZE)
LOR_IN_DIM = (SIZE, 1)
LOR_DIM = (LORS, SLICES, HEIGHT, LENGTH)
GT_DIM = (PHANTOMS, LORS)
LOR_CONV_DIM = (LORS, 1, SLICES, HEIGHT, LENGTH)

# The momentum factor, gamma.
MOMT = 0.95 #0.95 originally
