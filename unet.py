""" 
`unet.py` contains the network implementation for a UNet network according to the hyperparameters specified in `hyperparameters.py`. It saves the output system matrix and the trained model once at the end, but saves the loss throughout. 
"""

# Import necessary libraries.
import numpy as np
import torch
from torch import nn 
import pickle
import os
from torch.utils.data import TensorDataset, DataLoader
# Import other files.
import config as c
import hyperparameters as h
import databuilder

# Start time if required.
if h.TIMING_ == True:
    import time
    start_time = time.time()
    time_dict = {}

# Check for GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
torch.cuda.empty_cache()

# Set up the output files appropriately.
loss_fp = "results/loss_unet.pkl"
model_fp = "results/model_unet.pth"
sm_fp = "results/unet_sm.bin"
if os.path.exists(loss_fp):
    os.remove(loss_fp)
if os.path.exists(model_fp):
    os.remove(model_fp)
if os.path.exists(sm_fp):
    os.remove(sm_fp)
open(loss_fp, "wb").close()
print(f"Files cleared.")
torch.cuda.empty_cache()

# Load up the data.
lor = databuilder.lor.to(torch.float32)

# Add an extra dimension of 1 to the data.
lor = torch.reshape(lor, c.LOR_DIM)
lor = torch.unsqueeze(lor, dim=1)
# Move the lor to the cuda.
lor = lor.to(device)
# Scale the system matrix by 1000 so that it updates properly.
lor = torch.mul(lor, 1000)

# Move the data to the cpu.
input = lor.cpu().numpy()

# Add padding to the data to reshape it into even dimensions, so that it is easy to pass through the UNet network.
input = np.pad(input, h.PAD_WIDTH)
input = torch.from_numpy(input)

# Move the phantoms and the groundtruth to the cuda and ensure they are the right type.
phantom = torch.reshape(databuilder.phantom, (c.PHAN_RESHAPE)).to(torch.float32)
phantom = phantom.to(torch.float32)
phantom = torch.t(phantom)
phantom = phantom.to(device)
gt = databuilder.gt.to(torch.float32)
gt = gt.to(device)

# Scale the groundtruth by 1000 to match the input sm data.
gt = torch.mul(gt, 1000)

# Define our network.
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=1)
        self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride = 1)
        self.conv5 = nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=1, stride=1)
        
        self.upconv5 = nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=2, stride=2)
        
        self.conv6 = nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv7 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv8 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv9 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=1)

        self.conv10 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=1)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=h.DROP_OUT)

    # Define the forward pass.
    def forward(self, x):
        x.to(device)
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(self.dropout(self.maxpool(conv1_out))))
        conv3_out = self.relu(self.conv3(self.dropout(self.maxpool(conv2_out))))
        conv4_out = self.relu(self.conv4(self.dropout(self.maxpool(conv3_out))))
        conv5_out = self.relu(self.conv5(self.dropout(self.maxpool(conv4_out))))
        
        upconv5_out = self.relu(self.upconv5(conv5_out))
        # Skip connection.
        conv6_in = torch.cat((upconv5_out, conv4_out), dim = 1)
        conv6_out = self.relu(self.conv6(conv6_in))
        upconv4_out = self.relu(self.upconv4(conv6_out))
        # Skip connection.
        conv7_in = torch.cat((upconv4_out, conv3_out), dim=1)
        conv7_out = self.relu(self.conv7(conv7_in))
        upconv3_out = self.relu(self.upconv3(conv7_out))
        # Skip connection.
        conv8_in = torch.cat((upconv3_out, conv2_out), dim = 1)
        conv8_out = self.relu(self.conv8(conv8_in))
        upconv2_out = self.relu(self.upconv2(conv8_out))
        # Skip connection.
        conv9_in = torch.cat((upconv2_out, conv1_out), dim=1)
        conv9_out = self.relu(self.conv9(conv9_in))
        upconv1_out = self.relu(torch.abs((self.upconv1(conv9_out))))
        return upconv1_out

# Initialize our network. 
model = UNet()
model = model.to(device)

# Create the dataloaders
train_data = TensorDataset(input, gt)
train_loader = DataLoader(train_data, h.BATCH_SZ, shuffle=True)

# Define the loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=h.LRN_RT, momentum = c.MOMT)
if h.REGULARIZATION_ == True:
    optimizer = torch.optim.SGD(model.parameters(), lr=h.LRN_RT, momentum = c.MOMT, weight_decay = h.REG)

# Create loss dictionary
loss_dict = {}

# Define what happens during each iteration.
def iteration(count, ddata):
    # Clear the gradients of all optimized variables.
    optimizer.zero_grad()
    # Forward pass: compute predicted outputs by passing inputs to the model.
    ddata = ddata.to(device)
    output = model(ddata)
    fwd_project = torch.reshape(output, (h.BATCH_SZ, 1, 32, 96, 96))
    # Remove the padding from the output.
    fwd_project = fwd_project[:, :, 0:c.SLICES, 0:c.HEIGHT, 0:c.LENGTH]
    fwd_project = torch.squeeze(fwd_project, 1)
    fwd_project = torch.reshape(fwd_project, h.BATCH_RESHAPE)

    # Forward project the output of the encoding and decoding structure by performing matrix multiplication with the output and the phantom.
    fwd_project = torch.matmul(fwd_project, phantom)

    # Calculate the batch loss
    loss = criterion(fwd_project, target)

    # Backpropagation.
    loss.backward()

    # Access gradients and perform parameter updates (gradient descent).
    optimizer.step()

    # Update loss dictionary.
    if count in loss_dict:
        loss_dict[count] += loss
    else:
        loss_dict[count] = loss

    # Return the loss so that we can track it.
    return loss

# Track when the training starts.
if h.TIMING_ == True:
    start_training_time = time.time()
    print(f"Time elapsed so far: {start_training_time - start_time}")
    end_epoch_time = start_training_time
    end_count_time = start_training_time
print(f"Training starting now.")

# Run this training loop if thresholding is on.
if h.THRESHOLD_ == True:
    # Initialize the loss to something greater than the threshold value.
    looped_loss = h.THRES + 1
    # Keep track of the number of counts/epochs.
    count = 0
    # Set the model in training mode.
    model.train()
    # The training loop will run until the loss is less than or equal to the specified threshoold.
    while looped_loss > h.THRESH:
        # Keep track of which data is being plugged into the network currently, as we will iterate through all the data every epoch.
        inner_loop = 0
        for ddata, target in train_loader:
            # Move the data and the ground truth target to the cuda.
            ddata = ddata.to(device)
            target = target.to(device)
            # Update the loss to see if another epoch is needed or not while calling our iteration.
            looped_loss = iteration(count, ddata)
            # Update the loss dictionary.
            with open(loss_fp, "wb") as fp:
                pickle.dump(loss_dict, fp)
            # Keep track of timing if timing is on.
            if h.TIMING_ == True: 
                prev_epoch_time = end_epoch_time
                end_epoch_time = time.time()
                print(f"Epoch {inner_loop}, {count} took {end_epoch_time - prev_epoch_time} seconds.")
            # Increment the inner counter.
            inner_loop += 1

        # Save the model after each epoch in case of crashes or other issues.
        torch.save(model, model_fp)

        # Increment the counter.
        count += 1

    # Keep track of timing if it is on.
    if h.TIMING_ == True:
        prev_count_time = end_count_time
        end_count_time = time.time()
        print(f"Count time: {end_count_time - prev_count_time}")

# Run this training loop if thresholding is off.
else:
    # Set the model in training mode.
    model.train()
    for count in range(h.NUM_EPOCHS):
        # Keep track of counts/epochs.
        print(f"Count {count} started")
        # Keep track of which data is being plugged into the network currently, as we will iterate through all the data every epoch.
        inner_loop = 0
        for ddata, target in train_loader:
            # Move the data and the ground truth target to the cuda.
            ddata = ddata.to(device)
            target = target.to(device)
            # Update the loss for storage later.
            looped_loss = iteration(count, ddata)
           # Update the loss dictionary.
            with open(loss_fp, "wb") as fp:
                pickle.dump(loss_dict, fp)
            # Keep track of timing if timing is on.
            if h.TIMING_ == True: 
                prev_epoch_time = end_epoch_time
                end_epoch_time = time.time()
                print(f"Epoch {inner_loop}, {count} took {end_epoch_time - prev_epoch_time} seconds.")
            # Increment inner count.
            inner_loop += 1
        
        # Save the model after each epoch in case of crashes or other issues.
        torch.save(model, model_fp)

    # Keep track of timing if it is on.
    if h.TIMING_ == True:
        prev_count_time = end_count_time
        end_count_time = time.time()
        print(f"Count time: {end_count_time - prev_count_time}")

# Set the model in evaluation mode.
model.eval()

# Turn off gradient tracking.
for param in model.parameters():
    param.requires_grad = False

# Initialize the tensor to hold the trained system matrix.
final_output = torch.empty(c.LOR_DIM)

# Keep track of timing if it is turned on.
if h.TIMING_ == True:
    start_extract_tm = time.time()
    end_count_time = start_extract_tm

# Initialize index for extracting the trained system matrix.
i = 0
# This loop iterates through all of the lines of responses, passing them through the model and saving the output to the appropriate spot in `final_output` in order to extract the trained system matrix.
while i < 4096:
    with torch.no_grad():
        # We will iterate through all of the lines of response in batches.
        print(f"{i} to {i + h.BATCH_SZ}")

        # Extract the slice of the input data we are working with, and move it to the cuda.
        curr_slice = input[i:i + h.BATCH_SZ, :, :, :, :]
        curr_slice = curr_slice.to(device)

        # Pass the current slice through the model.
        curr_output = model(curr_slice)
        
        # Reshape and forward project the output, in the same way as was done in the function `iteration`.
        fwd_project = torch.reshape(curr_output, (h.BATCH_SZ, 1, 32, 96, 96))
        fwd_project = fwd_project[:, :, 0:c.SLICES, 0:c.HEIGHT, 0:c.LENGTH]
        fwd_project = torch.squeeze(fwd_project, 1)
        fwd_project = torch.reshape(fwd_project, h.BATCH_DIM)

        # Save the output to the `final_output` tensor.
        final_output[i : i + h.BATCH_SZ, :, :, :] = fwd_project

        # Increment the counter by the batch size.
        i += h.BATCH_SZ

        # Save the output as progress is made in case of crashing and other issues.
        torch.save(final_output, sm_fp)
        torch.cuda.empty_cache()

        # Keep track of timing if it is turned on.
        if h.TIMING_ == True:
            prev_count_time = end_count_time
            end_count_time = time.time()
            diff = end_count_time - prev_count_time
            print(f"Count time: {diff} sec")

# Scale the final_output by 0.001 to account for multiplying by 1000 in the beginning, before training.
final_output = torch.mul(final_output, 0.001)
# Save the output.
torch.save(final_output, sm_fp)
print(f"Training completed")

# Keep track of timing if it is turned on.
if h.TIMING_ == True:
    ending_time = time.time()
    extract = ending_time - start_extract_tm
    print(f"Total for matrix extraction in min: {extract / 60}")
    total = ending_time - start_time
    print(f"Total time in min: {total / 60}")




