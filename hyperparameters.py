"""
`hyperparameters.py` contains all of the hyperparameters that we change frequently. Unlike `config.py`, these variables are frequently changed in order to investigate their different effects on the results of the training network. As a result, they have been extracted to this separate file.
"""
# Import necessary libraries.
import config as c

# Turns timing on and off in our network.
TIMING_ = True

# Turns regularization on and off in our network.
REGULARIZATION_ = True
# Our regularization constant.
REG = 0.001

# Turns thresholding on and off in our network.
THRESHOLD_ = False
# Our threshold value; all values below this number are set to zero in our trained system matrix.
THRESH = 5.0

# Our dropout probability. Set to `0.0` in order to turn dropout off.
DROP_OUT = 0.01

# The learning rate of our model.
LRN_RT =  0.00000000001425 #10e-7 originally

# Number of iterations for training
NUM_EPOCHS = 10

# Batch size
BATCH_SZ = 16

# Dimensions:
BATCH_DIM = (BATCH_SZ, c.SLICES, c.LENGTH, c.HEIGHT)
BATCH_RESHAPE = (BATCH_SZ, c.SIZE)
BATCH_FLATTEN = (1, BATCH_SZ * c.SIZE)


# The specific amount of padding we will add to our input data to make it easier to pass through the convolutional network.
PAD_WIDTH = ((0, 0), (0, 0), (0, 7), (0, 21), (0, 21))

# The list of the epochs where we want to save the matrix output in `unet_save_multiple.py`
DISP_LST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# We append 4096, which is the code for the fully trained system matrix after all epochs have been completed.
DISP_LST.append(4096)
