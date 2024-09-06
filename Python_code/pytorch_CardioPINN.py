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
from loss_function import CardioLoss

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

#define activation function
class MySwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(30*x)
    

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # Initialize weights using Xavier (Glorot) initialization
        init.xavier_uniform_(m.weight)
        
        # Initialize biases to zero (I am not sure if the biases are updated in the tensorflow code)
        if m.bias is not None:
            init.zeros_(m.bias)
    
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,num_hidden):
        super(PINN, self).__init__()
        
        # Define the 10 hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_hidden)])
        
        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Use the custom activation function
        self.activation = MySwish()

        # Initialize weights and biases
        self.apply(initialize_weights)
    
    def forward(self, x):
        # Pass through each hidden layer with the custom activation function
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Pass through the output layer
        x = self.output_layer(x)
        return x
    

# Anatomical data
endo_fiber_angle =  np.pi/3  # helix angle at endocardium [rad] (used to be np.pi/3)
epi_fiber_angle  = -np.pi/3 # helix angle at epicardium [rad]
gamma_angle      = - 65.0 # orientation sheets [deg]
max_act          = 0.85e5 # maximum actuation stress value [Pa]

d_param          = 2  # number of points for tensor sampling of tuples (p_endo,T_a)
num_epochs = 2

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
        param_grid.append([p_range[ip],act_range[ia]])
param_grid = np.array(param_grid)

model = PINN(2,10,10,5).to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    total_loss = 0  # To track the loss for the entire epoch

    for i in range(len(param_grid[:,0])):
        input_sample = torch.tensor(param_grid[i], dtype=torch.float32).unsqueeze(0).to(device) # Get one training sample (add batch dimension)
        optimizer.zero_grad()  # Clear gradients for the current sample

        # Forward pass
        output = model(input_sample)

        # Compute the custom loss
        loss = CardioLoss(model,input_sample,vtk_file, HogdenHol,Fiber_params)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()  # Accumulate the loss for tracking

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(param_grid[:,0])}")