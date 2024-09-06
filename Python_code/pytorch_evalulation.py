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
import loss_function as loss

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

input_dim = 2
hidden_dim = 10
output_dim = 10
num_hidden = 5

# Circulation parameters
Windkessel_R  = 50.0/2.5  # systemic circulation resistance
Windkessel_C  = 5.0e-6/2.5 # systemic circulation compliance
end_diastolic_LV_pressure = 15.0  # end diastolic left ventricular pressure value
end_systolic_LV_pressure = 100.0 # end systolic left ventricular pressure value
diastolic_aortic_pressure = 45.0  # end diastolic aortic pressure value

# Constant values for all simulations
systole_length            = 250. # length of systole [ms]
diastole_length           = 650. # length of diastole [ms] (before 650)
dt_                       = 5.0  # time step [ms] (only to determine number of iterations)
n_steps = int((systole_length + diastole_length)/dt_) #number of time steps in simulation
t       = np.linspace(0,diastole_length+systole_length,n_steps) #list of time steps for which simulation is done
offset_time_ = diastole_length + systole_length/2.0
pressure_normalization = 150.0 # scaling value for pressure [mmHg]
stress_normalization   = 0.1e6 # scaling value for actuation stresses [Pa]
max_act          = 0.85e5 # maximum actuation stress value [Pa]

POD_folder_4D = "/data.lfpn/ibraun/Code/Cardio-PINN/Functional_model"
vtk_file = "/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/LV_mean_human_ESV.vtk"
out_folder = "test"
# Create an instance of the model
model = loss.PINN(input_dim, hidden_dim, output_dim,num_hidden)  # Replace with your model class

# Load the state dictionary into the model
model.load_state_dict(torch.load('model_weights.pth'))

# Set the model to evaluation mode if you're using it for inference
model.eval()

PHI,n_modesU,amplitude_range = dc.LoadPODmodes_FunctionalModel(POD_folder_4D,output_dim)
amplitude_max = loss.calculate_mode_normalisation(output_dim,amplitude_range)
a_pred = model(torch.tensor([0.0,0.0], dtype=torch.float32))
a_new = np.multiply(amplitude_max.detach().cpu().numpy(),a_pred.detach().cpu().numpy()) #predicte using NN for given pressure and active stress
volume = loss.Compute_Volume(a_new,vtk_file) #compute the volume of the new shape

print(f"Predicted EDV: {volume}")


# if not os.path.exists(out_folder + '/Simulation_results/'):
#     os.makedirs(out_folder + '/Simulation_results/')

# active_stress = [0]*n_steps #action stress in diastole

# for i in range(n_steps):
#     if t[i]>diastole_length and t[i]<=diastole_length + systole_length:   #during systole
#         active_stress[i] = max_act*(1.0-(2*(t[i]-offset_time_)/systole_length)**2) #behaves like a parabolar opened downward 

# active_stress = np.array(active_stress)
# ejection       = True

# pressure_LV = [0]*n_steps
# volume      = [0]*n_steps
# systolic_phase = False  #start simulation in diastole

# ED_id = int(diastole_length/dt_)-1
# a_out = np.zeros((n_steps,output_dim)) #matrix of calculated amplitudes for each the step in the simulation
# syst_steps = 0

# PHI,n_modesU,amplitude_range = dc.LoadPODmodes_FunctionalModel(POD_folder_4D,output_dim)
# amplitude_max = loss.calculate_mode_normalisation(output_dim,amplitude_range)

# for i in range(0,n_steps): #go through all steps in the simulation
#     print('Solving time-step: ',str(i) ,' of ',str(n_steps))

#     if pressure_LV[i-1]>diastolic_aortic_pressure: #initiate systolic phase
#         systolic_phase = True

#     if not systolic_phase:
#         if pressure_LV[i-1]<=end_diastolic_LV_pressure and active_stress[i] <= 0:
#             print('Diastolig filling phase')
#             max_volume = volume[i-1]
#             if i ==0 :
#                 pressure_LV[i] = 0.0 #start simulation at 0 pressure
#             else:
#                 pressure_LV[i] = pressure_LV[i-1] + end_diastolic_LV_pressure/diastole_length*(t[i]-t[i-1]) #increase the pressure linearly in diastolic filling
#         else :
#             print('.... Isovolumetric contraction')
#             pressure_LV[i] = loss.Isovolumetric_PressureUpdate(max_volume,active_stress[i]) #update the pressure while keeping the volumne constant
#     else: # when in systole
#         if ejection: #when in systolic ejection
#             print('.... Systole')
#             syst_steps+=1
#             deltaV = volume[i-1] - volume[i-2] 
#             pressure_LV[i] = loss.PressureUpdateSystole(active_stress[i]) #update pressure accordingly
#             if syst_steps>3: #after three steps in systole
#                 #if pressure_LV[i] < end_systolic_LV_pressure and pressure_LV[i-1] > pressure_LV[i]: #if the volumne is increasing or if the volumne is smaller than the volumne at the beginning of diastole
#                 if deltaV > 0 or volume[i-1]<=volume[0] + 2.:
#                     ejection = False #change to isovolumetric relaxation 
#         else:
#             print('.... Isovolumetric relaxation')
#             pressure_LV[i] = loss.Isovolumetric_PressureUpdate(save_volume,active_stress[i])  #update the pressure while keeping the volumne constant

#     a_new = np.multiply(amplitude_max,model([[pressure_LV[i]/pressure_normalization,active_stress[i]/stress_normalization]])) #predicte using NN for given pressure and active stress
#     a_out[i,:] = a_new #save calculated amplitudes for this simulation step
#     volume[i] = loss.Compute_Volume(a_new) #compute the volume of the new shape
#     save_volume = volume[i]
#     print('.... Pressure LV: '+ str(pressure_LV[i]) + ' mmHg, V: '+ str(volume[i]) + ' mL, Actuation strain: '+str(active_stress[i]/1e3)+' kPa', 'Index'+ str(i))

# plt.scatter(volume,pressure_LV, color = "black")
# plt.plot(volume,pressure_LV) #plot ans save LV loop
# plt.savefig(out_folder + f'/pV.png')
# plt.close()

# df = pd.DataFrame({'volume':volume, 'pressure_LV':pressure_LV})
# df.to_csv(out_folder + f'/P_volumes.csv', index=False)