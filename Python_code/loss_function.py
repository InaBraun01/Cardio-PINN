import torch
import numpy as np
import DeepCardioFunctions as dc
import monkey_functions as monkey

#search for device on which calculation will be done
device = (  
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def calculate_normal_direction(fx_s,fy_s,fz_s, sx_s,sy_s,sz_s):
    nx_s = fy_s*sz_s-fz_s*sy_s 
    ny_s = fz_s*sx_s-fx_s*sz_s
    nz_s = fx_s*sy_s-fy_s*sx_s

    nx = torch.tensor(nx_s,dtype=torch.float32).to(device)
    ny = torch.tensor(ny_s,dtype=torch.float32).to(device)
    nz = torch.tensor(nz_s,dtype=torch.float32).to(device)
    return nx,ny,nz


def calculate_mode_normalisation(n_modesU,amplitude_range):

    # Define normalization values for amplitudes of the bases
    amplitude_max = np.zeros((1,n_modesU))
    for i in range(n_modesU):
        #for each mode save the maximal amplitude in the range
        if abs(amplitude_range[i,0])> abs(amplitude_range[i,1]):
            amplitude_max[0,i] = amplitude_range[i,0]
        else:
            amplitude_max[0,i] = amplitude_range[i,1]  
    
    return torch.tensor(amplitude_max,dtype=torch.float32).to(device)

def FM_contribution_coordinates(PHI, n_points):
    Phix_s = PHI[0:n_points,:]            # FM contribution to x coordinate
    Phiy_s = PHI[n_points:2*n_points,:]   # FM contribution to y coordinate
    Phiz_s = PHI[2*n_points:3*n_points,:] # FM contribution to z coordinate

    return Phix_s, Phiy_s, Phiz_s


def compute_displacement(a_pred2, PHI, n_points):
    Phix_s, Phiy_s, Phiz_s = FM_contribution_coordinates(PHI, n_points)

    Phix = torch.tensor(Phix_s.T,dtype=torch.float32).to(device)
    Phiy = torch.tensor(Phiy_s.T,dtype=torch.float32).to(device)
    Phiz = torch.tensor(Phiz_s.T,dtype=torch.float32).to(device)

    ux = torch.matmul(a_pred2, Phix)
    uy = torch.matmul(a_pred2, Phiy)
    uz = torch.matmul(a_pred2, Phiz)
    return ux, uy, uz

def calculate_F_derivates(PHI, n_points,dFdx_s, dFdy_s, dFdz_s):
    Phix_s, Phiy_s, Phiz_s = FM_contribution_coordinates(PHI, n_points)

    dFudx_s = dFdx_s.dot(Phix_s)
    dFudy_s = dFdy_s.dot(Phix_s)
    dFudz_s = dFdz_s.dot(Phix_s)

    dFvdx_s = dFdx_s.dot(Phiy_s)
    dFvdy_s = dFdy_s.dot(Phiy_s)
    dFvdz_s = dFdz_s.dot(Phiy_s)

    dFwdx_s = dFdx_s.dot(Phiz_s)
    dFwdy_s = dFdy_s.dot(Phiz_s)
    dFwdz_s = dFdz_s.dot(Phiz_s)

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

    return dFudx, dFudy, dFudz, dFvdx, dFvdy, dFvdz, dFwdx, dFwdy, dFwdz

def compute_deformation_gradient(a_pred2, PHI,n_points,dFdx_s, dFdy_s, dFdz_s):

    dFudx, dFudy, dFudz, dFvdx, dFvdy, dFvdz, dFwdx, dFwdy, dFwdz = calculate_F_derivates(PHI, n_points,dFdx_s, dFdy_s, dFdz_s)

    F = {
        '00': torch.matmul(a_pred2, dFudx) + 1,
        '01': torch.matmul(a_pred2, dFudy),
        '02': torch.matmul(a_pred2, dFudz),
        '10': torch.matmul(a_pred2, dFvdx),
        '11': torch.matmul(a_pred2, dFvdy) + 1,
        '12': torch.matmul(a_pred2, dFvdz),
        '20': torch.matmul(a_pred2, dFwdx),
        '21': torch.matmul(a_pred2, dFwdy),
        '22': torch.matmul(a_pred2, dFwdz) + 1
    }
    return F

def compute_determinant(F):
    I3 = (F['00'] * F['11'] * F['22'] + F['01'] * F['12'] * F['20'] + F['02'] * F['10'] * F['21'] 
        - F['20'] * F['11'] * F['02'] - F['21'] * F['12'] * F['00'] - F['22'] * F['10'] * F['01'])
    J23 = torch.pow(I3, -2./3.)
    return I3, J23

def compute_cauchy_deformation_gradient(F, J23):
    C = {
        '00': (F['00']**2 + F['10']**2 + F['20']**2) * J23,
        '01': (F['00']*F['01'] + F['10']*F['11'] + F['20']*F['21']) * J23,
        '02': (F['00']*F['02'] + F['10']*F['12'] + F['20']*F['22']) * J23,
        '11': (F['01']**2 + F['11']**2 + F['21']**2) * J23,
        '12': (F['01']*F['02'] + F['11']*F['12'] + F['21']*F['22']) * J23,
        '22': (F['02']**2 + F['12']**2 + F['22']**2) * J23
    }
    return C

def compute_invariants(C, fx, fy, fz, sx, sy, sz, nx, ny, nz):
    I1 = C['00'] + C['11'] + C['22']
    I4f = fx*(C['00']*fx + C['01']*fy + C['02']*fz) + fy*(C['00']*fx + C['01']*fy + C['02']*fz) + fz*(C['00']*fx + C['01']*fy + C['02']*fz)
    I4s = sx*(C['00']*sx + C['01']*sy + C['02']*sz) + sy*(C['00']*sx + C['01']*sy + C['02']*sz) + sz*(C['00']*sx + C['01']*sy + C['02']*sz)
    I4n = nx*(C['00']*nx + C['01']*ny + C['02']*nz) + ny*(C['00']*nx + C['01']*ny + C['02']*nz) + nz*(C['00']*nx + C['01']*ny + C['02']*nz)
    I8fs = sx*(C['00']*fx + C['01']*fy + C['02']*fz) + sy*(C['00']*fx + C['01']*fy + C['02']*fz) + sz*(C['00']*fx + C['01']*fy + C['02']*fz)
    return I1, I4f, I4s, I4n, I8fs

def compute_passive_stress(I1, I3, I4f, I4s, I8fs, HogdenHol):
    Phi_passive = (HogdenHol.a_iso / 2. / HogdenHol.b_iso * (torch.exp(HogdenHol.b_iso * (I1-3.)) - 1.)
                + HogdenHol.a_f / 2. / HogdenHol.b_f * (torch.exp(HogdenHol.b_f * torch.pow(I4f-1., 2.)) - 1.)
                + HogdenHol.a_s / 2. / HogdenHol.b_s * (torch.exp(HogdenHol.b_s * torch.pow(I4s-1., 2.)) - 1.)
                + HogdenHol.k / 2. * torch.pow(I3-1., 2)
                + HogdenHol.a_fs / 2. / HogdenHol.b_fs * (torch.exp(HogdenHol.b_fs * torch.pow(I8fs, 2.0)) -1.))
    return Phi_passive

def compute_active_stress(I3, I4f, I4s, I4n, stress_normalization):
    Phi_active = stress_normalization / 2.0 / I3 * ((I4f - 1.0) + 0.3 * ((I4s -1.0) + (I4n - 1.0)))
    return Phi_active

def compute_total_stress(Phi_passive, Phi_active, stiff_scale, Nodal_volume, p_tf, E_sum_u):
    I_sum = stiff_scale * Phi_passive * Nodal_volume + p_tf[0][1] * Phi_active * Nodal_volume + E_sum_u
    return I_sum

def compute_external_pressure(ux, uy, uz, invF, Nodal_areax, Nodal_areay, Nodal_areaz, p_tf, pressure_normalization):
    newNodal_areax = invF['00']*Nodal_areax + invF['10']*Nodal_areay + invF['20']*Nodal_areaz
    newNodal_areay = invF['01']*Nodal_areax + invF['11']*Nodal_areay + invF['21']*Nodal_areaz
    newNodal_areaz = invF['02']*Nodal_areax + invF['12']*Nodal_areay + invF['22']*Nodal_areaz
    E_sum_u = p_tf[0][0] * pressure_normalization * 133.32 * (ux * newNodal_areax + uy * newNodal_areay + uz * newNodal_areaz)
    return E_sum_u

def calculate_nodal_area_volume(Nodal_area,Nodal_volume_s):
    Nodal_areax   = torch.tensor(Nodal_area[:,0],dtype=torch.float32).to(device)
    Nodal_areay   = torch.tensor(Nodal_area[:,1],dtype=torch.float32).to(device)
    Nodal_areaz   = torch.tensor(Nodal_area[:,2],dtype=torch.float32).to(device)
    Nodal_volume = torch.tensor(Nodal_volume_s[:,0],dtype=torch.float32).to(device)

    return Nodal_areax,Nodal_areay,Nodal_areaz,Nodal_volume

def create_torch_tensors(vari_list):
    modi_vari_list = []
    for var in vari_list:
        modi_vari_list.append(torch.tensor(var,dtype=torch.float32)).to(device)

    return modi_vari_list

def CardioLoss(model,p_tf,vtk_file, HogdenHol,Fiber_params,
            POD_folder_4D = "/data.lfpn/ibraun/Code/Cardio-PINN/Functional_model", #path to file in which POD components are stored
            n_modesU = 10, #number of POD components used to represent deformed geometry
            stiff_scale      = 0.75,   # scaling value of shear moduli of the material model [-] 
            pressure_normalization = 150.0, # scaling value for pressure [mmHg]
            stress_normalization   = 0.1e6 # scaling value for actuation stresses [Pa]
            ):
    
    Coords, Els, n_points,n_el, Node_par_coords ,Faces_Endo = monkey.LoadModelAnatomy(vtk_file)
    PHI,n_modesU,amplitude_range = dc.LoadPODmodes_FunctionalModel(POD_folder_4D,n_modesU)
    amplitude_max = calculate_mode_normalisation(n_modesU,amplitude_range)
    Nodal_area    = dc.GenerateNodalAreas(Faces_Endo,Coords)
    dFcdx_s, dFcdy_s, dFcdz_s, dFdx_s, dFdy_s, dFdz_s, Nodal_volume_s, Vol_el_s = dc.GradientOperator_AvgBased(Coords,Els,Node_par_coords)
    fx_s,fy_s,fz_s, sx_s,sy_s,sz_s = monkey.GenerateFibres(vtk_file,Fiber_params)
    fx = torch.tensor(fx_s,dtype=torch.float32).to(device)
    fy = torch.tensor(fy_s,dtype=torch.float32).to(device)
    fz = torch.tensor(fz_s,dtype=torch.float32).to(device)

    sx = torch.tensor(sx_s,dtype=torch.float32).to(device)
    sy = torch.tensor(sy_s,dtype=torch.float32).to(device)
    sz = torch.tensor(sz_s,dtype=torch.float32).to(device)
    Nodal_areax,Nodal_areay,Nodal_areaz,Nodal_volume = calculate_nodal_area_volume(Nodal_area,Nodal_volume_s)
    nx,ny,nz = calculate_normal_direction(fx_s,fy_s,fz_s, sx_s,sy_s,sz_s)

    a_pred = model(p_tf)
    a_pred2 = amplitude_max * a_pred
    ux, uy, uz = compute_displacement(a_pred2, PHI, n_points)
    F = compute_deformation_gradient(a_pred2, PHI,n_points,dFdx_s, dFdy_s, dFdz_s)
    I3, J23 = compute_determinant(F)
    C = compute_cauchy_deformation_gradient(F, J23)
    I1, I4f, I4s, I4n, I8fs = compute_invariants(C, fx, fy, fz, sx, sy, sz, nx, ny, nz)
    Phi_passive = compute_passive_stress(I1, I3, I4f, I4s, I8fs, HogdenHol)
    Phi_active = compute_active_stress(I3, I4f, I4s, I4n, stress_normalization)
    E_sum_u = compute_external_pressure(ux, uy, uz, F, Nodal_areax, Nodal_areay, Nodal_areaz, p_tf, pressure_normalization)
    I_sum = compute_total_stress(Phi_passive, Phi_active, stiff_scale, Nodal_volume, p_tf, E_sum_u)
    
    CardioEnergy = torch.sum(I_sum + E_sum_u)
    return CardioEnergy

