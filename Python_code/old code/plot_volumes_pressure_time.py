import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

def plot_function(volumes,pressure,index = None, save_title = False):

    if index != None:
        volumes = volumes[:index]
        pressure = pressure[:index]
    
    plt.scatter(volumes,pressure, color = "black")
    plt.plot(volumes,pressure, color = "grey")
    plt.xlabel("Volumes [ml]")
    plt.ylabel("Pressure [mmHg]")
    
    if save_title:
        plt.savefig(f"{save_title}")
    plt.show()


#In this code I plot the caculated volume and pressure against time as well as the calculated pv loop
#I plot it once over the entire cardiac cycle and once only until the ES state (part of the cycle which I tuned)

systole_length            = 250. # length of systole [ms]
diastole_length           = 650. # length of diastole [ms] (before 650)
dt_                       = 5.0  # time step [ms] (only to determine number of iterations)

n_steps = int((systole_length + diastole_length)/dt_) #number of time steps in simulation 

offset_time_ = diastole_length + systole_length/2.0

t       = np.linspace(0,diastole_length+systole_length,n_steps) #list of time steps for which simulation is done

df_pV = pd.read_csv('/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/EDP_25/P_volumes.csv')

plot_function(df_pV['bp_volume'], df_pV['pressure'], index = 170, save_title= "/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/EDP_25/Pv_loop.png" )
