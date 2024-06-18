'''
Training and simulation of a Physics Informed Neural Network for Cardiac Mechanics

Copiright:  Buoso Stefano 2021. ETH Zurich
            buoso@biomed.ee.ethz.ch
'''

import sys,os,shutil
import tensorflow.compat.v1 as tf #code was written for the older tensorflow version 1.10
tf.disable_v2_behavior()  #disable version 2 behavior of tensor flow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

import vtk
from   vtk.util.numpy_support import vtk_to_numpy

import DeepCardioFunctions as dc

import matplotlib as mpl

tf.compat.v1.set_random_seed(10)     #set seed

#devine activation function
def mySwish(x):
    return x*tf.nn.sigmoid(30*x)

#initialize using the Xavier Method
def initialize_NN(layers):  
    weights = []
    biases  = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        #draw weights from normal distribution 
        W = init_xavierMethod(size=[layers[l], layers[l+1]])
        #initialize biases with zero
        b = tf.zeros([1,layers[l+1]], dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases
        
def init_xavierMethod(size):
    in_dim  = size[0]
    out_dim = size[1] 
    #use normal Xavier initialisation       
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32) #default of mean is set to zero
    
    
def neural_net(X, weights, biases):

    num_layers = len(weights) + 1
    H =  X
    for l in range(0,num_layers-2):
        #in every layer except last use swish activation function
        W = weights[l]
        b = biases[l]
        H = mySwish(tf.add(tf.matmul(H,W), b))
    #dont use activation function in last layer
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H,W), b)
    return Y

def CardioLoss():

    a_pred2 = amplitude_max*a_pred # a_pred: predicted amplitudes, amplitude_max: amplitudes to scale with 
    # Compute nodal displacements
    ux = tf.matmul(a_pred2,Phix) 
    uy = tf.matmul(a_pred2,Phiy) 
    uz = tf.matmul(a_pred2,Phiz) 
    # Compute deformation gradient
    DefGradient00 = tf.matmul(a_pred2,dFudx) + 1 #adding the 1 because F= grad(u) + 1
    DefGradient01 = tf.matmul(a_pred2,dFudy)
    DefGradient02 = tf.matmul(a_pred2,dFudz)

    DefGradient10 = tf.matmul(a_pred2,dFvdx)
    DefGradient11 = tf.matmul(a_pred2,dFvdy) + 1 #adding the 1 because F= grad(u) + 1
    DefGradient12 = tf.matmul(a_pred2,dFvdz)

    DefGradient20 = tf.matmul(a_pred2,dFwdx)
    DefGradient21 = tf.matmul(a_pred2,dFwdy)
    DefGradient22 = tf.matmul(a_pred2,dFwdz) + 1 #adding the 1 because F= grad(u) + 1

    # Compute determinant of deformation gradient, which is in the paper defined as J
    I3 = DefGradient00*DefGradient11*DefGradient22 + DefGradient01*DefGradient12*DefGradient20 + DefGradient02*DefGradient10*DefGradient21 \
       - DefGradient20*DefGradient11*DefGradient02 - DefGradient21*DefGradient12*DefGradient00 - DefGradient22*DefGradient10*DefGradient01

    J23 = tf.pow(I3,-2./3.)

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
    Phi_passive       =   HogdenHol.a_iso/2./HogdenHol.b_iso*( tf.exp( HogdenHol.b_iso*       (I1-3.)   ) - 1.) \
                      +   HogdenHol.a_f/2./HogdenHol.b_f    *( tf.exp( HogdenHol.b_f  *tf.pow(I4f-1.,2.)) - 1.) \
                      +   HogdenHol.a_s/2./HogdenHol.b_s    *( tf.exp( HogdenHol.b_s  *tf.pow(I4s-1.,2.)) - 1.) \
                      +   HogdenHol.k/2.*tf.pow(I3-1.,2) \
                      +   HogdenHol.a_fs/2./HogdenHol.b_fs*(tf.exp(HogdenHol.b_fs*tf.pow(I8fs,2.0)) -1.) 

    # Compute active stress contribution,in the equqtion set mu = 0.3
    Phi_active     = stress_normalization/2.0/I3 *( (I4f - 1.0) + 0.3*( (I4s -1.0) + (I4n - 1.0) ) ) # stress normalization: scaling value for actuation stresses [Pa]

    # Compute total stresses in myocardium, integration apprx as multiplying times volume
    I_sum          = stiff_scale*Phi_passive*Nodal_volume \
                   + p_tf[0][1]*Phi_active*Nodal_volume  #p_tf[0][1] is the actuation stress, stiff scale : scaling value of shear moduli of the material model [-]

    # Compute contribution of external pressure loading on endocardium
    newNodal_areax = invF_00*Nodal_areax + invF_10*Nodal_areay + invF_20*Nodal_areaz  #F inverse times normal on endocardium
    newNodal_areay = invF_01*Nodal_areax + invF_11*Nodal_areay + invF_21*Nodal_areaz
    newNodal_areaz = invF_02*Nodal_areax + invF_12*Nodal_areay + invF_22*Nodal_areaz
    E_sum_u       = p_tf[0][0]*pressure_normalization*133.32*(ux*newNodal_areax + uy*newNodal_areay +uz*newNodal_areaz)   #133.32 area of the faces ??

    # Compute total cost function
    CardioEnergy  = tf.reduce_sum(I_sum + E_sum_u)
	
    return CardioEnergy

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
    while err > 0.001 and iterations < 10:
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


# Input section
local_path    = os.getcwd()
cases_folder  = local_path + '/Synthetic_shapes/'
case_name     = 'Shape1'
mesh_name     = 'Anatomy.vtk'
POD_folder_4D = local_path  + '/Functional_model/'
out_folder    = cases_folder + case_name + '/PINN_data/'

# Anatomical data
endo_fiber_angle =  40.0  # helix angle at endocardium [deg]
epi_fiber_angle  = - 60.0 # helix angle at epicardium [deg]
gamma_angle      = - 65.0 # orientation sheets [deg]
max_act          = 0.85e5 # maximum actuation stress value [Pa]
stiff_scale      = 0.75   # scaling value of shear moduli of the material model [-] 

# Circulation parameters
Windkessel_R  = 50.0   # systemic circulation resistance
Windkessel_C  = 5.0e-6 # systemic circulation compliance
end_diastolic_LV_pressure = 15.0  # end diastolic left ventricular pressure value
diastolic_aortic_pressure = 45.0  # end diastolic aortic pressure value

# Constant values for all simulations
systole_length            = 250. # length of systole [ms]
diastole_length           = 650. # length of diastole [ms]
dt_                       = 5.0  # time step [ms] (only to determine number of iterations)

# Network architecture parameters
n_input_variables = 2  # number of input variables
n_modesU          = 10 # number of functional bases as last layer
hidden_layers     = 5  # number of hidden layers
hidden_neurons    = 10 # number of neurons per hidden layer
pressure_normalization = 150.0 # scaling value for pressure [mmHg]
stress_normalization   = 0.1e6 # scaling value for actuation stresses [Pa]

epochs           = 30 # number of training epocs (in the paper used 300 epochs)
d_param          = 20  # number of points for tensor sampling of tuples (p_endo,T_a)  
learn_rate       = 0.0001 # learning rate

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# Material model From 
#     Sack KL et al. (2018) Construction and Validation of Subject-Specific 
#             Biventricular Finite-Element Models of Healthy and Failing Swine 
#             Hearts From High-Resolution DT-MRI. Front. Physiol. 9:539.
#             doi: 10.3389/fphys.2018.00539
a_iso = tf.constant(1.05e3,dtype=np.float32)
b_iso = tf.constant(7.52,dtype=np.float32)
a_f   = tf.constant(3.465e3,dtype=np.float32)
b_f   = tf.constant(14.472,dtype=np.float32)
a_s   = tf.constant(0.481e3,dtype=np.float32)
b_s   = tf.constant(12.548,dtype=np.float32)
a_fs  = tf.constant(0.283e3,dtype=np.float32)
b_fs  = tf.constant(3.088,dtype=np.float32)
Bulk  = tf.constant(10.5e5,dtype=np.float32)

# Determine classes for material models parameters and fiber orientations
HogdenHol       = dc.matParameters(a_iso, b_iso, a_f, b_f, a_s, b_s, a_fs, b_fs,Bulk)
Fiber_params    = dc.class_FibersData(endo_fiber_angle,epi_fiber_angle,0,0,gamma_angle)

# Read anatomy and parametrization
print('. Reading reference parametric anatomy')
Coords, Els, n_points,n_el, Node_par_coords, e_t, e_l, e_c ,Faces_Endo = dc.LoadModelAnatomy(cases_folder + case_name + '/' + mesh_name)

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

Phix_s = PHI[0:n_points,:]            # FM contribution to x coordinate
Phiy_s = PHI[n_points:2*n_points,:]   # FM contribution to y coordinate
Phiz_s = PHI[2*n_points:3*n_points,:] # FM contribution to z coordinate


# Generate microsctructure
fx_s,fy_s,fz_s, sx_s,sy_s,sz_s = dc.GenerateFibers(e_t,e_l,e_c,Node_par_coords,Fiber_params)
dc.WriteFibers2VTK(Coords,Els, fx_s,fy_s,fz_s, sx_s,sy_s,sz_s, out_folder+'/GeneratedMicrostructure.vtk')

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

# Generate constant tf variables for network
Coords_x = tf.constant(Coords[:,0],dtype=np.float32)
Coords_y = tf.constant(Coords[:,1],dtype=np.float32)
Coords_z = tf.constant(Coords[:,2],dtype=np.float32)

fx = tf.constant(fx_s,dtype=np.float32)
fy = tf.constant(fy_s,dtype=np.float32)
fz = tf.constant(fz_s,dtype=np.float32)

sx = tf.constant(sx_s,dtype=np.float32)
sy = tf.constant(sy_s,dtype=np.float32)
sz = tf.constant(sz_s,dtype=np.float32)

nx_s = fy_s*sz_s-fz_s*sy_s 
ny_s = fz_s*sx_s-fx_s*sz_s
nz_s = fx_s*sy_s-fy_s*sx_s

nx = tf.constant(nx_s,dtype=np.float32)
ny = tf.constant(ny_s,dtype=np.float32)
nz = tf.constant(nz_s,dtype=np.float32)

Phix = tf.constant(Phix_s.T,dtype=np.float32)  
Phiy = tf.constant(Phiy_s.T,dtype=np.float32)
Phiz = tf.constant(Phiz_s.T,dtype=np.float32)

Nodal_areax   = tf.constant(Nodal_area[:,0],dtype=np.float32)
Nodal_areay   = tf.constant(Nodal_area[:,1],dtype=np.float32)
Nodal_areaz   = tf.constant(Nodal_area[:,2],dtype=np.float32)

Nodal_volume = tf.constant(Nodal_volume_s[:,0],dtype=np.float32)

dFdx = tf.constant(dFdx_s.T,dtype=np.float32) #This is because I am computing
dFdy = tf.constant(dFdy_s.T,dtype=np.float32) # u.T = (Phi*a).T = a.T * Phi.T
dFdz = tf.constant(dFdz_s.T,dtype=np.float32)

dFudx = tf.constant(dFudx_s.T,dtype=np.float32) 
dFudy = tf.constant(dFudy_s.T,dtype=np.float32) 
dFudz = tf.constant(dFudz_s.T,dtype=np.float32)

dFvdx = tf.constant(dFvdx_s.T,dtype=np.float32) 
dFvdy = tf.constant(dFvdy_s.T,dtype=np.float32) 
dFvdz = tf.constant(dFvdz_s.T,dtype=np.float32)

dFwdx = tf.constant(dFwdx_s.T,dtype=np.float32) 
dFwdy = tf.constant(dFwdy_s.T,dtype=np.float32)
dFwdz = tf.constant(dFwdz_s.T,dtype=np.float32)

# Define network
layers = [n_input_variables] #collect size of all layers of NN
for hs in range(hidden_layers):
     layers.append(hidden_neurons)
layers.append(n_modesU) 
weights, biases = initialize_NN(layers)  #initialize weights for all layers of NN

# Define input placeholder for inpu parameters (pressure,Ta)
p_tf = tf.placeholder(tf.float32, shape=[None,n_input_variables]) #variable to which data is assigned later

print('. Building network')

# Generate (p_endo,T_a) tuples for training
p_range           = np.linspace(0.0,1.0,d_param)
act_range         = np.linspace(0.0,1.0,d_param)
param_grid = [] #create grid of all combinations of parameters to test
for ip in range(d_param): # allowed pressure values
    for ia in range(d_param):
        param_grid.append([p_range[ip],act_range[ia]])
param_grid = np.array(param_grid)

a_pred = neural_net(p_tf, weights, biases) #calculate amplitudes based on the input parameters

loss      = CardioLoss()
optimiser = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)

init_op = tf.global_variables_initializer()
saver   = tf.train.Saver()

n_steps = int((systole_length + diastole_length)/dt_) #number of time steps in simulation
t       = np.linspace(0,diastole_length+systole_length,n_steps) #list of time steps for which simulation is done

offset_time_ = diastole_length + systole_length/2.0

loss_vector = [0]*epochs
with tf.Session() as sess:  #session holds values of intermediate results and variables
    # initialise the variables
    sess.run(init_op)  #initializes variables before use not used in tensorflow v2
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(param_grid.shape[0]): 
            _,c = sess.run([optimiser, loss],feed_dict={p_tf:[param_grid[i,:]]}) #run optimiser and loss for all combinations of input parameters
            avg_cost += c    #sum up the loss for all different inputs -> not really averaged
        print("Epoch:", (epoch + 1), "cost =", str(avg_cost))
        loss_vector[epoch] = avg_cost
    plt.plot(loss_vector[5:epoch]) #plot loss over the different epochs
    plt.tight_layout()
    plt.savefig(out_folder +'/Loss_function.png',dpi=400) # save the plot
    plt.close()

    save_path = saver.save(sess, out_folder+'/Trained_model.ckpt') #save the session of the NN
    print("Model saved in path: %s" % save_path)

# with tf.Session() as sess:
#     saver.restore(sess, out_folder+'/Trained_model.ckpt')
    # Run case 1
    for csel in range(1,2):  #I really don't understand why the run through this loop 7 times. I think this loop can be getten rid off

        if not os.path.exists(out_folder + '/Simulation_results/'):
            os.makedirs(out_folder + '/Simulation_results/')

        active_stress = [0]*n_steps #action stress in diastole

        for i in range(n_steps):
            if t[i]>diastole_length and t[i]<=diastole_length + systole_length:   #during systole
                active_stress[i] = max_act*(1.0-(2*(t[i]-offset_time_)/systole_length)**2) #behaves like a parabolar opened downward 

        active_stress = np.array(active_stress)
        ejection       = True

        pressure_LV = [0]*n_steps
        volume      = [0]*n_steps
        systolic_phase = False  #start simulation in diastole

        ED_id = int(diastole_length/dt_)-1
        a_out = np.zeros((n_steps,n_modesU)) #matrix of calculated amplitudes for each the step in the simulation
        syst_steps = 0

        for i in range(0,n_steps): #go through all steps in the simulation
            print('Solving time-step: ',str(i) ,' of ',str(n_steps))

            if pressure_LV[i-1]>diastolic_aortic_pressure: #initiate systolic phase
                systolic_phase = True

            if not systolic_phase:
                if pressure_LV[i-1]<=end_diastolic_LV_pressure and active_stress[i] <= 0:
                    print('Diastolig filling phase')
                    max_volume = volume[i-1]
                    if i ==0 :
                        pressure_LV[i] = 0.0 #start simulation at 0 pressure
                    else:
                        pressure_LV[i] = pressure_LV[i-1] + end_diastolic_LV_pressure/diastole_length*(t[i]-t[i-1]) #increase the pressure linearly in diastolic filling
                else :
                    print('.... Isovolumetric contraction')
                    pressure_LV[i] = Isovolumetric_PressureUpdate(max_volume,active_stress[i]) #update the pressure while keeping the volumne constant
            else: # when in systole
                if ejection: #when in systolic ejection
                    print('.... Systole')
                    syst_steps+=1
                    deltaV = volume[i-1] - volume[i-2] 
                    pressure_LV[i] = PressureUpdateSystole(active_stress[i]) #update pressure accordingly
                    if syst_steps>3: #after three steps in systole
                        if deltaV > 0 or volume[i-1]<=volume[0]+2.: #if the volumne is increasing or if the volumne is smaller than the volumne at the beginning of diastole
                            ejection = False #change to isovolumetric relaxation 
                else:
                    print('.... Isovolumetric relaxation')
                    pressure_LV[i] = Isovolumetric_PressureUpdate(volume[0],active_stress[i])  #update the pressure while keeping the volumne constant

            a_new = np.multiply(amplitude_max,sess.run(a_pred, feed_dict={p_tf:[[pressure_LV[i]/pressure_normalization,active_stress[i]/stress_normalization]]})) #predicte using NN for given pressure and active stress
            a_out[i,:] = a_new #save calculated amplitudes for this simulation step
            volume[i] = Compute_Volume(a_new) #compute the volume of the new shape

            print('.... Pressure LV: '+ str(pressure_LV[i]) + ' mmHg, V: '+ str(volume[i]) + ' mL, Actuation strain: '+str(active_stress[i]/1e3)+' kPa')

        plt.plot(volume,pressure_LV) #plot ans save LV loop
        plt.savefig(out_folder + '/pV.png')
        plt.close()

        ccdumb = 0
        for i in range(0,n_steps,2):
            #for i simulation step calculate the displacement vector and new coordinates

            disp_x = a_out[i,:].dot(Phix_s.T) #Phix_s contribution of FM in x direction
            disp_y = a_out[i,:].dot(Phiy_s.T)
            disp_z = a_out[i,:].dot(Phiz_s.T)

            NewCoordsx =  disp_x.copy() + Coords[:,0].copy()
            NewCoordsy =  disp_y.copy() + Coords[:,1].copy()
            NewCoordsz =  disp_z.copy() + Coords[:,2].copy()


            if ccdumb<10:
                outFile = open(out_folder + '/Simulation_results/Displ_0'+str(ccdumb)+'.vtk','w')
            else:
                outFile = open(out_folder+ '/Simulation_results/Displ_'+str(ccdumb)+'.vtk','w')
            #Write shape of LV in vtk file
            outFile.write('# vtk DataFile Version 4.0\n')
            outFile.write('vtk output\n')
            outFile.write('ASCII\n')
            outFile.write('DATASET UNSTRUCTURED_GRID \n')
            outFile.write('POINTS '+str(Coords.shape[0])+' float\n')
            for j in range(Coords.shape[0]): #write coordinates in file
                # outFile.write(str(Coords[j,0])+' ')  #write original coordinates in vtk file
                # outFile.write(str(Coords[j,1])+' ') #write original coordinates in vtk file
                # outFile.write(str(Coords[j,2])+' ') #write original coordinates in vtk file

                outFile.write(str(NewCoordsx[j])+' ')  #write updated coordinates in vtk file
                outFile.write(str(NewCoordsy[j])+' ')#write updated coordinates in vtk file
                outFile.write(str(NewCoordsz[j])+' ')#write updated coordinates in vtk file
                outFile.write('\n')
            outFile.write( 'CELLS ' + str( Els.shape[0] ) + ' ' + str( (Els.shape[0]) * 5 ) )
            outFile.write('\n')
            for k in range( Els.shape[0] ): #write nodes in each cells in file
                outFile.write( '4 ' )
                for j in range( 4 ):
                    outFile.write( str( Els[k,j]) + ' ' )
                outFile.write('\n')
            # write cell types
            outFile.write( '\n\nCELL_TYPES ' + str( Els.shape[0] ) )
            for k in range( Els.shape[0] ):
                outFile.write( '\n10' )
            outFile.write('\nPOINT_DATA '+str(Coords.shape[0])+'\n')
            outFile.write('VECTORS displ float \n')
            for k in range(Coords.shape[0]):
                outFile.write('%1.4f %1.4f %1.4f \n' %(disp_x[k],disp_y[k],disp_z[k]) ) #write displacement for each node
            outFile.close()
            ccdumb+=1

        ff = open(out_folder + '/P_volumes.txt','w') #save pressure and volumn throughout simulation
        for i in range(len(volume)):
            ff.write('%1.4f ' %volume[i])
            ff.write('0.0 ')
            ff.write('%1.4f ' %pressure_LV[i])
            ff.write('\n')
        ff.close()

ff = open(out_folder+'/Newtork_architecture.py','w') #save the architecture of the NN

ff.write('n_modesU          = %d\n' %n_modesU)
ff.write('hidden_neurons    = %d\n' %hidden_neurons)
ff.write('hidden_layers     = %d\n' %hidden_layers)
ff.write('n_input_variables = %d\n' %n_input_variables)

ff.write('pressure_normalization = %1.4f\n' %pressure_normalization)
ff.write('stress_normalization   = %1.4f\n' %stress_normalization)
ff.close()



#test 
# save the original coordinates and see if the displacement field can be displayed in paraview
# then save the new coordinates and see if you can see the left ventricle move overtime

#run Bobo with only 5 modes (with old hyperparams) => output calculated modes and create mesh for time frame zero using Generate_shapes.py, compare with output from mesh generation
#run PINN on generated synthetic mesh
#look at pv loop and loss function
#compare generated meshes from MRI with generated meshes from PINN