"""
`display.py` contains functions which allow different types of data to be displayed, including loss and updated system matrices.
"""

# Import necessary libraries.
import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
import databuilder
import config as c

# `display_pickled_loss` displays the loss of a network over epochs given the  `.pkl` filepath where the loss dictionary is stored.
def display_pickled_loss(fp, name):
    with open(fp, "rb") as fp:
        loss_dict = pickle.load(fp)
    for entry in loss_dict:
        loss_dict[entry] = loss_dict[entry].sum().item()
    keys = loss_dict.keys()
    vals = loss_dict.values()
    print(f"loss_dict : {loss_dict}")

    # Plotting
    plt.plot(keys, vals)
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title(name)
    plt.show()

# `display_sum_LoRs` displays the sum of the lines of responses before training and after training, with and without masking, and sometimes with absolute value applied, given the filepath of the updated system matrix.
def display_sum_LoRs(fp):
    
    # Display sum of lors before training.
    lor_sum = np.empty(c.DIM)
    first_lor = torch.reshape(databuilder.lor, (c.LOR_DIM))
    second_lor = first_lor.numpy()
    for line in range(second_lor.shape[0]):
        lor_sum += second_lor[line, :, :, :]
    slice_16 = lor_sum[17, :, :]
    plt.imshow(slice_16, cmap='hot', )
    plt.colorbar()
    plt.title("Sum of LoRs before training - Slice 17")
    plt.show()

    # Display the sum after training, without mask.
    with open(fp, "rb") as file:
        new_lor = torch.load(file)
    new_lor = new_lor.detach()
    lor_sum = np.empty(c.DIM)
    new_lor = torch.reshape(new_lor, (c.LOR_DIM))
    first_lor = new_lor.numpy()
    for line in range(first_lor.shape[0]):
        curr_slice = first_lor[line, :, :, :]
        lor_sum = curr_slice + lor_sum
    slice_17 = lor_sum[17, :, :]
    plt.imshow(slice_17, cmap='hot')
    plt.colorbar()
    plt.title("Sum of LoRs after training - Slice 17")
    plt.show()

    # Display the absolute value of the sum after training.
    lor_sum = np.empty(c.DIM)
    abs_lor = torch.abs(new_lor).numpy()
    for line in range(abs_lor.shape[0]):
        lor_sum = abs_lor[line, :, :, :] + lor_sum
    slice_17 = lor_sum[17, :, :]
    plt.imshow(slice_17, cmap='hot', )
    plt.colorbar()
    plt.title("Sum of LoRs after training - Slice 17 \n Abs Val")
    plt.show()


# `display_new_LoR` displays the old, untrained system matrix and then the new, trained system matrix given the filepath of the new, trained system matrix as well as the specific line of response and slice to display.
def display_new_LoR(fp, LoR, slice):

    # Load the new system matrix.
    with open(fp, "rb") as file:
        new_lor = torch.load(file)
    print(f"new_lor size: {new_lor.shape}")
    new_lor = torch.reshape((new_lor.detach()), c.LOR_DIM).numpy()

    # Display the new system matrix for the specified slice.
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    slice_new = new_lor[LoR, slice, :, :]
    slice_old = databuilder.og_lor[LoR, slice, :, :]
    im1 = axes[0].imshow(slice_old, cmap='hot')
    cbar1 = fig.colorbar(im1, ax=axes[0])
    axes[0].set_title(f"Old LoR {LoR}, Slice {slice}")
    im2 = axes[1].imshow(slice_new, cmap='hot')
    cbar2 = fig.colorbar(im2, ax=axes[1])
    axes[1].set_title(f"New LoR {LoR}, Slice {slice}")
    plt.show()

    # Display the new system matrix with absolute value applied.
    slice_new_abs = np.abs(slice_new)
    plt.imshow(slice_new_abs, cmap='hot')
    plt.colorbar()
    plt.title(f'Slice Number {slice} \n New lor, Abs Val')
    plt.show()

#`display_sm` simply displays the given 2D torch tensor as an image.
def display_sm(fp, ver):
    with open(fp, "rb") as file:
        im = torch.load(file)
    print(f" im size: {im.shape}")
    plt.imshow(im.numpy(), cmap="hot")
    plt.colorbar()
    plt.title(f"LoR 3870, Slice 15 after Epoch {ver}")
    plt.show()

# `boosted.py` boosts the background of the slice to be displayed in order to check if the system matrix has any interesting features in the background.
def boosted(fp, LoR, slice):
    # Load the new LoR.
    with open(fp, "rb") as file:
        new_lor = torch.load(file)
    # Extract the specific slice according to the input arguments LoR and slice.
    new_lor = torch.reshape((new_lor.detach()), c.LOR_DIM).numpy()
    slice_17_new = new_lor[LoR, slice, :, :]
    m = np.max(slice_17_new)
    plt.imshow(slice_17_new, cmap = 'hot')
    plt.title("Boosted background slice 17- New LoR")
    # Limit the scale so that the background is boosted.
    plt.clim(0, m - .013)
    plt.colorbar()
    plt.show()