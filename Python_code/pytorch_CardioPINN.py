import sys,os,shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init

import numpy as np
import matplotlib.pyplot as plt
import math 

import vtk
from   vtk.util.numpy_support import vtk_to_numpy

import DeepCardioFunctions as dc
import monkey_functions as test

import matplotlib as mpl
import pandas as pd
from loss_function import CardioLoss,PINN,prequi_CardioLoss

torch.manual_seed(10)  #set a seed
#search for device on which calculation will be done
device = (  
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Anatomical data
endo_fiber_angle =  np.pi/3  # helix angle at endocardium [rad] (used to be np.pi/3)
epi_fiber_angle  = -np.pi/3 # helix angle at epicardium [rad]
gamma_angle      = - 65.0 # orientation sheets [deg]
max_act          = 0.85e5 # maximum actuation stress value [Pa]
input_dim = 2
hidden_dim = 10
output_dim = 10
num_hidden = 5

d_param= 20  # number of points for tensor sampling of tuples (p_endo,T_a)
num_epochs = 400

a_iso = torch.tensor(151.75323017591577,dtype=torch.float32)
b_iso = torch.tensor(2.389951971229547,dtype=torch.float32)
a_f   = torch.tensor(307.13640608445553,dtype=torch.float32)
b_f   = torch.tensor(4.140143426252412,dtype=torch.float32)
a_s   = torch.tensor(159.69495336552495,dtype=torch.float32)
b_s   = torch.tensor(2.4212905561925377,dtype=torch.float32)
a_fs  = torch.tensor(39.5830423438744,dtype=torch.float32)
b_fs  = torch.tensor(0.572786454863574,dtype=torch.float32)
Bulk  = torch.tensor(10.5e5,dtype=torch.float32)

# Determine classes for material models parameters and fiber orientations
HogdenHol       = dc.matParameters(a_iso, b_iso, a_f, b_f, a_s, b_s, a_fs, b_fs,Bulk)
Fiber_params    = dc.class_FibersData(endo_fiber_angle,epi_fiber_angle,0,0,gamma_angle)
vtk_file = "/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/LV_mean_human_ESV.vtk"

# Generate (p_endo,T_a) tuples for training
p_range           = np.linspace(0.0,1.0,d_param)
act_range         = np.linspace(0.0,1.0,d_param)
param_grid = [] #create grid of all combinations of parameters to test
for ip in range(d_param): # allowed pressure values
    for ia in range(d_param):
        #param_grid.append([p_range[ip],act_range[ia]])
        param_grid.append([0.0,0.0])
param_grid = np.array(param_grid)

model = PINN(input_dim, hidden_dim, output_dim,num_hidden).to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Preparing tensors for the loss function")
amplitude_max,fx, fy, fz, sx, sy, sz, nx, ny, nz, PHI,n_points,dFdx_s, dFdy_s, dFdz_s,Nodal_areax,Nodal_areay,Nodal_areaz,Nodal_volume = prequi_CardioLoss(vtk_file,Fiber_params)
# Wrap your CardioLoss with torch.vmap to process batches
batched_loss_fn = torch.vmap(lambda input_vector: CardioLoss(model, input_vector, vtk_file, HogdenHol, Fiber_params,amplitude_max,fx, fy, fz, sx, sy, sz, nx, ny, nz, PHI,n_points,dFdx_s, dFdy_s, dFdz_s,Nodal_areax,Nodal_areay,Nodal_areaz,Nodal_volume))
    
batch_size = 400  # Define your batch size here
loss_vector = [0]*num_epochs
print("Starting the training")
for epoch in range(num_epochs):
    total_loss = 0  # To track the loss for the entire epoch
    for i in range(0, len(param_grid[:, 0]), batch_size):
        # Get a batch of input samples
        input_batch = torch.tensor(param_grid[i:i + batch_size], dtype=torch.float32).unsqueeze(0).to(device)
        
        optimizer.zero_grad()  # Clear gradients for the current batch
        
        # Forward pass (this is vectorized across the batch)
        output_batch = model(input_batch)
        
        # Compute the custom loss over the batch using vmap
        loss_batch = batched_loss_fn(input_batch)
        
        # Sum the losses across the batch
        total_loss_batch = loss_batch.sum()
        
        # Backward pass
        total_loss_batch.backward()

        # Update weights
        optimizer.step()

        total_loss += total_loss_batch.item()

    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(param_grid[:,0])}")
    loss_vector[epoch] = total_loss/len(param_grid[:,0])
    
plt.plot(loss_vector[5:epoch]) #plot loss over the different epochs
plt.tight_layout()
plt.savefig('Loss_function.png',dpi=400) # save the plot
plt.show()


torch.save(model.state_dict(), 'model_weights.pth')

#print(f"This took {}s.")