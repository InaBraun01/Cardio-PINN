'''
Training and simulation of a Physics Informed Neural Network for Cardiac Mechanics

Copiright:  Buoso Stefano 2021. ETH Zurich
            buoso@biomed.ee.ethz.ch
'''

import sys,os,shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

def CardioLoss(model,p_tf):
    a_pred = model(p_tf)
    a_pred2 = amplitude_max*a_pred # a_pred: predicted amplitudes, amplitude_max: amplitudes to scale with 
    # Compute nodal displacements
    ux = torch.matmul(a_pred2,Phix) 
    uy = torch.matmul(a_pred2,Phiy) 
    uz = torch.matmul(a_pred2,Phiz) 
    # Compute deformation gradient
    DefGradient00 = torch.matmul(a_pred2,dFudx) + 1 #adding the 1 because F= grad(u) + 1
    DefGradient01 = torch.matmul(a_pred2,dFudy)
    DefGradient02 = torch.matmul(a_pred2,dFudz)

    DefGradient10 = torch.matmul(a_pred2,dFvdx)
    DefGradient11 = torch.matmul(a_pred2,dFvdy) + 1 #adding the 1 because F= grad(u) + 1
    DefGradient12 = torch.matmul(a_pred2,dFvdz)

    DefGradient20 = torch.matmul(a_pred2,dFwdx)
    DefGradient21 = torch.matmul(a_pred2,dFwdy)
    DefGradient22 = torch.matmul(a_pred2,dFwdz) + 1 #adding the 1 because F= grad(u) + 1

    # Compute determinant of deformation gradient, which is in the paper defined as J
    I3 = DefGradient00*DefGradient11*DefGradient22 + DefGradient01*DefGradient12*DefGradient20 + DefGradient02*DefGradient10*DefGradient21 \
       - DefGradient20*DefGradient11*DefGradient02 - DefGradient21*DefGradient12*DefGradient00 - DefGradient22*DefGradient10*DefGradient01

    J23 = torch.pow(I3,-2./3.)

    # Compute Left Cauchy deformation gradient components, which is J(-2/3) times C in paper 
    C00 = (DefGradient00*DefGradient00 + DefGradient10*DefGradient10 + DefGradient20*DefGradient20)*J23
    C01 = (DefGradient00*DefGradient01 + DefGradient10*DefGradient11 + DefGradient20*DefGradient21)*J23
    C02 = (DefGradient00*DefGradient02 + DefGradient10*DefGradient12 + DefGradient20*DefGradient22)*J23

    C10 = (DefGradient01*DefGradient00 + DefGradient11*DefGradient10 + DefGradient21*DefGradient20)*J23
    C11 = (DefGradient01*DefGradient01 + DefGradient11*DefGradient11 + DefGradient21*DefGradient21)*J23
    C12 = (DefGradient01*DefGradient02 + DefGradient11*DefGradient12 + DefGradient21*DefGradient22)*J23

    C20 = (DefGradient02*DefGradient00 + DefGradient12*DefGradient10 + DefGradient22*DefGradient20)*J23
    C21 = (DefGradient02*DefGradient01 + DefGradient12*DefGradient11 + DefGradient22*DefGradient21)*J23
    C22 = (DefGradient02*DefGradient02 + DefGradient12*DefGradient12 + DefGradient22*DefGradient22)*J23

    # Compute inverse of deformation gradient
    invF_00 =   DefGradient11*DefGradient22 - DefGradient21*DefGradient12 # I am not dividing by I3 since I would
    invF_10 = - DefGradient10*DefGradient22 + DefGradient20*DefGradient12 # neet to multiply by it in the calculation
    invF_20 =   DefGradient10*DefGradient21 - DefGradient20*DefGradient11 # fof the deformed area for components of E_sum_u

    invF_01 = - DefGradient01*DefGradient22 + DefGradient21*DefGradient02
    invF_11 =   DefGradient00*DefGradient22 - DefGradient20*DefGradient02
    invF_21 = - DefGradient00*DefGradient21 + DefGradient20*DefGradient01

    invF_02 =   DefGradient01*DefGradient12 - DefGradient11*DefGradient02
    invF_12 = - DefGradient00*DefGradient12 + DefGradient10*DefGradient02
    invF_22 =   DefGradient00*DefGradient11 - DefGradient10*DefGradient01

    # Compute invariants of Left Cauchy deformation gradient, constants in equation 2 in paper 
    I1 = C00 + C11 + C22

    I4f  = fx*(C00*fx+C01*fy+C02*fz) + fy*(C10*fx+C11*fy+C12*fz) + fz*(C20*fx+C21*fy+C22*fz)
    I4s  = sx*(C00*sx+C01*sy+C02*sz) + sy*(C10*sx+C11*sy+C12*sz) + sz*(C20*sx+C21*sy+C22*sz)
    I4n  = nx*(C00*nx+C01*ny+C02*nz) + ny*(C10*nx+C11*ny+C12*nz) + nz*(C20*nx+C21*ny+C22*nz)

    I8fs = sx*(C00*fx+C01*fy+C02*fz) + sy*(C10*fx+C11*fy+C12*fz) + sz*(C20*fx+C21*fy+C22*fz)

    # Compute passive stress contribution, W_p equation 1 in paper, om 4th line should it be i3 to pwer of 2 ??????????
    Phi_passive       =   HogdenHol.a_iso/2./HogdenHol.b_iso*( torch.exp( HogdenHol.b_iso*       (I1-3.)   ) - 1.) \
                      +   HogdenHol.a_f/2./HogdenHol.b_f    *( torch.exp( HogdenHol.b_f  *torch.pow(I4f-1.,2.)) - 1.) \
                      +   HogdenHol.a_s/2./HogdenHol.b_s    *( torch.exp( HogdenHol.b_s  *torch.pow(I4s-1.,2.)) - 1.) \
                      +   HogdenHol.k/2.*torch.pow(I3-1.,2) \
                      +   HogdenHol.a_fs/2./HogdenHol.b_fs*(torch.exp(HogdenHol.b_fs*torch.pow(I8fs,2.0)) -1.) 

    # Compute active stress contribution,in the equqtion set mu = 0.3
    Phi_active     = stress_normalization/2.0/I3 *( (I4f - 1.0) + 0.3*( (I4s -1.0) + (I4n - 1.0) ) ) # stress normalization: scaling value for actuation stresses [Pa]

    # Compute total stresses in myocardium, integration apprx as multiplying times volume
    I_sum          = stiff_scale*Phi_passive*Nodal_volume \
                   + p_tf[1]*Phi_active*Nodal_volume  #p_tf[1] is the actuation stress, stiff scale : scaling value of shear moduli of the material model [-]

    # Compute contribution of external pressure loading on endocardium
    newNodal_areax = invF_00*Nodal_areax + invF_10*Nodal_areay + invF_20*Nodal_areaz  #F inverse times normal on endocardium
    newNodal_areay = invF_01*Nodal_areax + invF_11*Nodal_areay + invF_21*Nodal_areaz
    newNodal_areaz = invF_02*Nodal_areax + invF_12*Nodal_areay + invF_22*Nodal_areaz
    E_sum_u       = p_tf[0]*pressure_normalization*133.32*(ux*newNodal_areax + uy*newNodal_areay +uz*newNodal_areaz)   #133.32 area of the faces ??

    # Compute total cost function
    CardioEnergy  = torch.sum(I_sum + E_sum_u)
	
    return CardioEnergy

def Batch_CardioLoss(model, input_batch):
    individual_losses = []
    for input_vector in input_batch:
        loss = CardioLoss(model, input_vector)
        individual_losses.append(loss)
    return torch.stack(individual_losses)

def Isovolumetric_PressureUpdate(volume_constraint,active_s):
    ''' Iterative scheme for the calculation of the pressure value to preserve the volumetric constrain
    during the isovolumetric phase
    Iteratively determine the value p_0 for the given active stress active_s and while preserving the volume
    '''

    dp = 10/133.32#Pa   
    iterations = 0
    err = 1.
    p_0 = pressure_LV[i-1]  #pressure due to bloof pool
    #iterativelu update the pressure until the volume has deviated too much or max iterations are reached
    while err > 0.001 and iterations <  50:
        iterations += 1
        #using NN calculate a_prep for the current pressure and activation stress and scale a_prep to get appropriate predicted amplitudes
        a_0 = np.multiply(amplitude_max,sess.run(a_pred, feed_dict={p_tf:[[p_0/pressure_normalization,active_s/stress_normalization]]}))
        #using NN calculate a_prep for the updated pressure and current activation stress and scale a_prep to get appropriate predicted amplitudes
        a_1 = np.multiply(amplitude_max,sess.run(a_pred, feed_dict={p_tf:[[(p_0+dp)/pressure_normalization,active_s/stress_normalization]]}))

        #calculate volumne for both calculated sets of amplitudes
        V_lv_0 = Compute_Volume(a_0)
        V_lv_1 = Compute_Volume(a_1)

        v_error = V_lv_0-volume_constraint
        
        local_compliance = (V_lv_1-V_lv_0)/dp

        deltap = - (v_error)/local_compliance
        err = abs(deltap)   #calculate change in volume due to change in pressure 
        p_0 += deltap   #update pressure
        max_iterations = iterations

    print(f"MAX ITERATIONS: {max_iterations}")

    return p_0

def Compute_Volume(a_sel):

    ''' Compute left ventricular blood pool volume'''
    #calculate displacements in all directions
    disp_x = a_sel[0,:].dot(Phix_s.T) 
    disp_y = a_sel[0,:].dot(Phiy_s.T)
    disp_z = a_sel[0,:].dot(Phiz_s.T)
    #update the coordinates with the calculated displacement
    NewCoords = np.concatenate((Coords[:,0].reshape(-1,1)+disp_x.reshape(-1,1),Coords[:,1].reshape(-1,1)+disp_y.reshape(-1,1),Coords[:,2].reshape(-1,1)+disp_z.reshape(-1,1)),1) #concatenate vectors horizontally
    
    volume_blood_ML = 0
    for j in range(len(Faces_Endo)): #for each face of the tetrahedral cells
        sel_el = Faces_Endo[j] #for all nodes on the face
        oa = np.array(NewCoords[sel_el[0],:]) #x,y,z coordinates of the node
        ob = np.array(NewCoords[sel_el[1],:]) #x,y,z coordinates of the node
        oc = np.array(NewCoords[sel_el[2],:]) #x,y,z coordinates of the node
        volume_blood_ML += 1.0/6.0*abs(np.dot(np.cross(oa,ob),oc))*1e6  #add together individual volumn parts

    return volume_blood_ML

def PressureUpdateSystole(active_s):
    ''' Iterative procedure to couple sistolic function with systemic circulation
    Calculate pressure in systolic phase using two elemnt windkessel model'''

    dp = 10. # Pa
    DT = (t[i] - t[i-1])/1e3
    iterations = 0
    err = 1.
    p_0 = pressure_LV[i-1]

    while err > 0.001 and iterations < 10:
        iterations += 1
        #using NN calculate a_prep for the current pressure and ypdated pressure and activation stress and scale a_prep to get appropriate predicted amplitudes
        a_0 = np.multiply(amplitude_max,sess.run(a_pred, feed_dict={p_tf:[[p_0/pressure_normalization,active_s/stress_normalization]]}))
        a_1 = np.multiply(amplitude_max,sess.run(a_pred, feed_dict={p_tf:[[(p_0+dp/133.32)/pressure_normalization,active_s/stress_normalization]]}))
        V_lv_0 = Compute_Volume(a_0)
        V_lv_1 = Compute_Volume(a_1)

        LV_compliance = (V_lv_1-V_lv_0)/dp # change in volumn compared to change in pressure

        residual_windkessel = -(volume[i-1]-V_lv_0)+ Windkessel_C *(p_0-pressure_LV[i-1])*133.32 + DT*p_0/Windkessel_R*133.32 #for the windkesselmodel this should be zero
        first_derivative    = LV_compliance + Windkessel_C  + DT/Windkessel_R #derivative of residual

        # NR iteration
        delta_p = - residual_windkessel/first_derivative/133.32 #iteratively change the pressure until the change in pressure is too large

        err = abs(delta_p)
        p_0 += delta_p

    return p_0

#Input file 
data_file = "Test_data/"
#vtk_file = "/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/Shape1/Anatomy_human.vtk"
vtk_file = "/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/LV_mean_human_ESV.vtk"

# Input section
# local_path    = os.getcwd()
# cases_folder  = local_path + '/Synthetic_shapes/'
# case_name     = "double_Maike_ED"
# POD_folder_4D = local_path  + '/Functional_model/'
# out_folder    = cases_folder + case_name + '/PINN_data/human'

POD_folder_4D = "/data.lfpn/ibraun/Code/Cardio-PINN/Functional_model"
# out_folder    = cases_folder + case_name + '/PINN_data/'


#Scale the input mesh inorder to change the unit at which the node positions are given from mm to m
#test.Scale_mesh(vtk_file, data_file, out_folder)


# Anatomical data
endo_fiber_angle =  np.pi/3  # helix angle at endocardium [rad] (used to be np.pi/3)
epi_fiber_angle  = -np.pi/3 # helix angle at epicardium [rad]
gamma_angle      = - 65.0 # orientation sheets [deg]
max_act          = 0.85e5 # maximum actuation stress value [Pa]
stiff_scale      = 0.75   # scaling value of shear moduli of the material model [-] 

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

# Network architecture parameters
n_input_variables = 2  # number of input variables
n_modesU          = 10 # number of functional bases as last layer
hidden_layers     = 5  # number of hidden layers
hidden_neurons    = 10 # number of neurons per hidden layer
pressure_normalization = 150.0 # scaling value for pressure [mmHg]
stress_normalization   = 0.1e6 # scaling value for actuation stresses [Pa]

epochs           = 300 # number of training epocs
d_param          = 20  # number of points for tensor sampling of tuples (p_endo,T_a)  
learn_rate       = 0.01 # learning rate


# Material model From 
#     Sack KL et al. (2018) Construction and Validation of Subject-Specific 
#             Biventricular Finite-Element Models of Healthy and Failing Swine 
#             Hearts From High-Resolution DT-MRI. Front. Physiol. 9:539.
#             doi: 10.3389/fphys.2018.00539

#OLD TUNED VALUES
num_a_iso = 151.75323017591577 
num_b_iso = 2.389951971229547
num_a_f   = 307.13640608445553
num_b_f   = 4.140143426252412
num_a_s   = 159.69495336552495
num_b_s   = 2.4212905561925377
num_a_fs  = 39.5830423438744
num_b_fs  = 0.572786454863574
num_Bulk  = 10.5e5


a_iso = torch.tensor(151.75323017591577,dtype=torch.float32).to(device)
b_iso = torch.tensor(2.389951971229547,dtype=torch.float32).to(device)
a_f   = torch.tensor(307.13640608445553,dtype=torch.float32).to(device)
b_f   = torch.tensor(4.140143426252412,dtype=torch.float32).to(device)
a_s   = torch.tensor(159.69495336552495,dtype=torch.float32).to(device)
b_s   = torch.tensor(2.4212905561925377,dtype=torch.float32).to(device)
a_fs  = torch.tensor(39.5830423438744,dtype=torch.float32).to(device)
b_fs  = torch.tensor(0.572786454863574,dtype=torch.float32).to(device)
Bulk  = torch.tensor(10.5e5,dtype=torch.float32).to(device)

# out_folder = f"/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/Diastolic_filling_Ta_{max_act}_2.5_CR_a_iso_{round(num_a_iso)}"
out_folder = f"/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/test_swine"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# Determine classes for material models parameters and fiber orientations
HogdenHol       = dc.matParameters(a_iso, b_iso, a_f, b_f, a_s, b_s, a_fs, b_fs,Bulk)
Fiber_params    = dc.class_FibersData(endo_fiber_angle,epi_fiber_angle,0,0,gamma_angle)

# Read anatomy and parametrization
print('. Reading reference parametric anatomy')
#Coords, Els, n_points,n_el, Node_par_coords, e_t, e_l, e_c ,Faces_Endo = dc.LoadModelAnatomy(vtk_file)
Coords, Els, n_points,n_el, Node_par_coords ,Faces_Endo = test.LoadModelAnatomy(vtk_file)

# Load functional model bases (FM)
PHI,n_modesU,amplitude_range = dc.LoadPODmodes_FunctionalModel(POD_folder_4D,n_modesU)

# Define normalization values for amplitudes of the bases
amplitude_max = np.zeros((1,n_modesU))
for i in range(n_modesU):
    #for each mode save the maximal amplitude in the range
    if abs(amplitude_range[i,0])> abs(amplitude_range[i,1]):
        amplitude_max[0,i] = amplitude_range[i,0]
    else:
        amplitude_max[0,i] = amplitude_range[i,1]    	
amplitude_max = torch.tensor(amplitude_max,dtype=torch.float32).to(device)

Phix_s = PHI[0:n_points,:]            # FM contribution to x coordinate
Phiy_s = PHI[n_points:2*n_points,:]   # FM contribution to y coordinate
Phiz_s = PHI[2*n_points:3*n_points,:] # FM contribution to z coordinate

# Generate microsctructure
#fx_s,fy_s,fz_s, sx_s,sy_s,sz_s = dc.GenerateFibers(e_t,e_l,e_c,Node_par_coords,Fiber_params) #fx_s: x coordinate of fibre direction for each node in numpy array
fx_s,fy_s,fz_s, sx_s,sy_s,sz_s = test.GenerateFibres(vtk_file,Fiber_params)
# fx_s,fy_s,fz_s = f_vector.T
# sx_s, sy_s,sz_s = s_vector.T

dc.WriteFibers2VTK(Coords,Els,fx_s,fy_s, fz_s, sx_s, sy_s,sz_s, out_folder+'/GeneratedMicrostructure.vtk')

# Generate Nodal area vector for the computation of boundary traction forces
Nodal_area    = dc.GenerateNodalAreas(Faces_Endo,Coords)
print('. Computing deformation gradient matrices')

# Generate gradient operator matrices
dFcdx_s, dFcdy_s, dFcdz_s, dFdx_s, dFdy_s, dFdz_s, Nodal_volume_s, Vol_el_s = dc.GradientOperator_AvgBased(Coords,Els,Node_par_coords)

dFudx_s = dFdx_s.dot(Phix_s)
dFudy_s = dFdy_s.dot(Phix_s)
dFudz_s = dFdz_s.dot(Phix_s)

dFvdx_s = dFdx_s.dot(Phiy_s)
dFvdy_s = dFdy_s.dot(Phiy_s)
dFvdz_s = dFdz_s.dot(Phiy_s)

dFwdx_s = dFdx_s.dot(Phiz_s)
dFwdy_s = dFdy_s.dot(Phiz_s)
dFwdz_s = dFdz_s.dot(Phiz_s)

# Generate constant torch variables for network
Coords_x = torch.tensor(Coords[:,0],dtype=torch.float32).to(device)
Coords_y = torch.tensor(Coords[:,1],dtype=torch.float32).to(device)
Coords_z = torch.tensor(Coords[:,2],dtype=torch.float32).to(device)

fx = torch.tensor(fx_s,dtype=torch.float32).to(device)
fy = torch.tensor(fy_s,dtype=torch.float32).to(device)
fz = torch.tensor(fz_s,dtype=torch.float32).to(device)

sx = torch.tensor(sx_s,dtype=torch.float32).to(device)
sy = torch.tensor(sy_s,dtype=torch.float32).to(device)
sz = torch.tensor(sz_s,dtype=torch.float32).to(device)

nx_s = fy_s*sz_s-fz_s*sy_s 
ny_s = fz_s*sx_s-fx_s*sz_s
nz_s = fx_s*sy_s-fy_s*sx_s

nx = torch.tensor(nx_s,dtype=torch.float32).to(device)
ny = torch.tensor(ny_s,dtype=torch.float32).to(device)
nz = torch.tensor(nz_s,dtype=torch.float32).to(device)

Phix = torch.tensor(Phix_s.T,dtype=torch.float32).to(device)
Phiy = torch.tensor(Phiy_s.T,dtype=torch.float32).to(device)
Phiz = torch.tensor(Phiz_s.T,dtype=torch.float32).to(device)

Nodal_areax   = torch.tensor(Nodal_area[:,0],dtype=torch.float32).to(device)
Nodal_areay   = torch.tensor(Nodal_area[:,1],dtype=torch.float32).to(device)
Nodal_areaz   = torch.tensor(Nodal_area[:,2],dtype=torch.float32).to(device)

Nodal_volume = torch.tensor(Nodal_volume_s[:,0],dtype=torch.float32).to(device)

dFdx = torch.tensor(dFdx_s.T,dtype=torch.float32).to(device) #This is because I am computing
dFdy = torch.tensor(dFdy_s.T,dtype=torch.float32).to(device) # u.T = (Phi*a).T = a.T * Phi.T
dFdz = torch.tensor(dFdz_s.T,dtype=torch.float32).to(device)

dFudx = torch.tensor(dFudx_s.T,dtype=torch.float32).to(device)
dFudy = torch.tensor(dFudy_s.T,dtype=torch.float32).to(device)
dFudz = torch.tensor(dFudz_s.T,dtype=torch.float32).to(device)

dFvdx = torch.tensor(dFvdx_s.T,dtype=torch.float32).to(device) 
dFvdy = torch.tensor(dFvdy_s.T,dtype=torch.float32).to(device) 
dFvdz = torch.tensor(dFvdz_s.T,dtype=torch.float32).to(device)

dFwdx = torch.tensor(dFwdx_s.T,dtype=torch.float32).to(device) 
dFwdy = torch.tensor(dFwdy_s.T,dtype=torch.float32).to(device)
dFwdz = torch.tensor(dFwdz_s.T,dtype=torch.float32).to(device)


print('. Building network')

# Generate (p_endo,T_a) tuples for training
p_range           = np.linspace(0.0,1.0,d_param)
act_range         = np.linspace(0.0,1.0,d_param)
param_grid = [] #create grid of all combinations of parameters to test
for ip in range(d_param): # allowed pressure values
    for ia in range(d_param):
        param_grid.append([p_range[ip],act_range[ia]])
param_grid = np.array(param_grid)

model = loss.PINN(n_input_variables, hidden_neurons, n_modesU ,hidden_layers).to(device)

loss_vector = []

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr = learn_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
batched_loss_fn = torch.vmap(lambda input_vector:  CardioLoss(model,input_vector))
    
batch_size = 128  # Define your batch size here
loss_vector = []
print("Starting the training")

loss_vector = []
for epoch in range(epochs):
    total_loss = 0  # To track the loss for the entire epoch
    total_sum_loss = 0
    for i in range(0, len(param_grid[:, 0]), batch_size):
        # Get a batch of input samples
        input_batch = torch.tensor(param_grid[i:i + batch_size], dtype=torch.float32, requires_grad=True).unsqueeze(0).to(device)
        
        optimizer.zero_grad()  # Clear gradients for the current batch
        
        # Forward pass (this is vectorized across the batch)
        output_batch = model(input_batch)
        
        # Compute the custom loss over the batch using vmap
        #loss_batch = batched_loss_fn(input_batch)
        loss_batch = Batch_CardioLoss(model, input_batch[0])
        
        # take the average of the losses across the batch
        mean_loss_batch = loss_batch.mean()
        sum_loss_batch = loss_batch.sum()
        
        # Backward pass
        mean_loss_batch.backward()

        # Update weights
        optimizer.step()

        total_loss += mean_loss_batch.item()
        total_sum_loss += sum_loss_batch.item()

    # At the end of each epoch, update the scheduler
    scheduler.step(mean_loss_batch)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")
    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{epochs}], Sum Loss: {total_sum_loss}, Average Loss: {total_loss}")
    loss_vector.append(total_loss)

df = pd.DataFrame({'volume':loss_vector})
df.to_csv("old_all_batch_loss.csv") 
torch.save(model.state_dict(), 'model_weights_batch.pth')

