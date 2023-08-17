"""
`unet_save_multiple_display.py` displays the results from `unet_save_multiple.py` using functions from `display.py`.
"""

# Import other files.
import display 
import hyperparameters as h

# The names of the files where our results are stored.
loss_fp = "results/loss_unet_save_multiple.pkl"
model_fp = "results/model_unet_save_multiple.pth"
# These file paths do not have their `.bin` endings because we will be viewing multiple versions of them throughout the epochs.
sm_fp = "results/unet_save_multiple_sm"
slice_fp = "results/lor3870slice15"

# Displays the loss in a line graph format.
display.display_pickled_loss(loss_fp, "Loss")

# Iterate through the display list to view line 3870, slice 15 in the trained system matrices after each epoch.
for ver in h.DISP_LST:
    curr_slice_fp = slice_fp + f"{ver}.bin"
    display.display_sm(curr_slice_fp, ver)

# Display the sum of the lines of responses before and after training.
final_sm_fp = sm_fp + f"4096.bin"
display.display_sum_LoRs(final_sm_fp)
