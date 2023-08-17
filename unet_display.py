"""
`unet_display.py` displays the results from `unet.py` using functions from `display.py`.
"""

# Import necessary libraries.
import torch
# Import other files.
import display 

# The names of the files where our results are stored.
loss_fp = "results/loss_unet.pkl"
model_fp = "results/model_unet.pth"
sm_fp = "results/unet_sm.bin"

# Displays the loss in a line graph format.
display.display_pickled_loss(loss_fp, "Loss")
# Displays the given line of response and slice of the new system matrix.
display.display_new_LoR(sm_fp, 3870, 15)


# Uncomment here to turn on cutoff, where entries with values less than 3% of the max value are set to 0. ->

"""
# Load the new, trained system matrix.
with open(sm_fp, "rb") as file:
        new_lor = torch.load(file)
# Define the cutoff to be 3% of the maximum value.
cutoff = 0.03 * (torch.max(new_lor))
# Reshape the new system matrix to allow for looping through it.
new_lor = torch.reshape((new_lor.detach()), c.LOR_DIM)

# Iterate through all of the elements in the new system matrix.
for line in range(c.LORS):
        for slice in range(c.SLICES):
                for h in range(c.HEIGHT):
                        for l in range(c.LENGTH):
                                # Extract entry value.
                                x = new_lor[line, slice, h, l]
                                # Check if the value is below the cutoff.
                                if x.item() <= cutoff:
                                        # If the value is below the cutoff, set it to 0.
                                        new_lor[line, slice, h, l] = torch.tensor(0)
                                else:
                                        pass
        print(f"LoR {line} finished")
torch.save(new_lor, "thresholded_unet_sm.bin")
# Displays the given line of response and slice of the thresholded system matrix.
display.display_new_LoR("thresholded_unet_sm.bin", 3870, 15)
"""
