import sys
import os
import vtk
import numpy as np
from volume_cal_functions import load_vtk_unstructured_grid, convert_to_polydata, calculate_volume, cap_bowl, create_solid_mesh, calculate_volume
import monkey_functions as monkey
from collections import defaultdict
from vtk.util.numpy_support import vtk_to_numpy
import pandas as pd

#CALCULATE VOLUME AS PREVIOUSLY DONE BY ME FROM THE MESHES
def cal_bp_volume(vtk_file_path):
    # Load the VTK file
    unstructured_grid = load_vtk_unstructured_grid(vtk_file_path)

    # Extract the surface from the unstructured grid (convert to polydata)
    surface_mesh = convert_to_polydata(unstructured_grid)

    # Cap the bowl to make it a closed surface
    closed_mesh = cap_bowl(surface_mesh)

    # Create a solid mesh from the closed surface
    solid_mesh = create_solid_mesh(closed_mesh)

    # Calculate the volume of the closted mesh (blood pool and heart muscle)
    volume = calculate_volume(solid_mesh)
    #print(f"The volume of the blood pool and heart muscle is: {volume} cubic units")

    #Calculate the volume of the heart muscle
    musc_volume = calculate_volume(surface_mesh)
    #print(f"The volume of the heart muscle is: {musc_volume} cubic units")

    #Calculate the volume of the blood pool
    bp_volume = volume - musc_volume
    #print(f"The volume of the heart muscle is: {bp_volume} cubic units")

    return bp_volume, musc_volume


#CALCULATE VOLUMES AS DONE IN PINN CODE

def Compute_Volume(vtk_file_path):
    #Problems might arise here more because the faces need to be nicely aligned for this to work
    #each edge needs to be shared by exactly two faces and all faces need to be oriented the same way for this to work
    #additionally the center of the system needs to be in the origin (0,0,0)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    data = reader.GetOutput()

    n_points = data.GetNumberOfPoints()   #number of points
    n_el     = data.GetNumberOfCells()    #number of cells
    Coords   =  vtk_to_numpy(data.GetPoints().GetData())   #coordinates of each point
    Els      = np.zeros((n_el,4),dtype=int)  #connections
    for i in range(n_el):
        cell_type = data.GetCellType(i)
        n_nodes_el   = data.GetCell(i).GetPointIds().GetNumberOfIds()    #number of vertices for each cell
        for n_sel in range(n_nodes_el):
            Els[i,n_sel] = int(data.GetCell(i).GetPointId(n_sel)) # save list of vertices contained in cell

    faces_connectivity = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])  #vertices connected on the 4 faces of the tetrahedron
    Faces_Endo = []  #list of points that make up tetrahedral cell with the one specific vertex
    start_faces = True
    for kk in range(n_el): #for each cell 
        el_points = Els[kk,:] #vertices found in kk cell, there are always 4 vertices in each cell
        for jj in range(4): #for all vertices making up the cell, save the connection of vertices on the faces
            if start_faces:
                Faces_Endo  = np.array(el_points[faces_connectivity[jj]],dtype=int).reshape(1,-1)
                start_faces = False
            else:
                Faces_Endo = np.concatenate((Faces_Endo,np.array(el_points[faces_connectivity[jj]],dtype=int).reshape(1,-1)),0)
   

    #Coords, Els, n_points,n_el, Node_par_coords ,Faces_Endo = monkey.LoadModelAnatomy(vtk_file_path)
    
    volume_blood_ML = 0
    for j in range(len(Faces_Endo)): #for each face of the tetrahedral cells
        sel_el = Faces_Endo[j] #for all nodes on the face
        oa = np.array(Coords[sel_el[0],:]) #x,y,z coordinates of the node
        ob = np.array(Coords[sel_el[1],:]) #x,y,z coordinates of the node
        oc = np.array(Coords[sel_el[2],:]) #x,y,z coordinates of the node
        volume_blood_ML += 1.0/6.0*abs(np.dot(np.cross(oa,ob),oc))*1e6  #add together individual volumn parts

    return volume_blood_ML


def calculate_volumes_PINN(directory_path):

    # List to store outputs

    df = pd.DataFrame(columns=['time_step', 'bp_volume', 'pressure'])

    df_PINN_400 = pd.read_csv(f"{directory_path}/P_volumes.csv")
    mesh_dir = f"{directory_path}/Simulation_results"

    # Loop through all files in the directory
    for file_name in os.listdir(mesh_dir):
        # Check if the file has a .vtk extension
        if file_name.endswith('.vtk'):
            print(file_name)
            time_step = file_name.split('_')[1].split('.')[0]
            # Full file path
            file_path = os.path.join(mesh_dir, file_name)

            # Process the file and store the output in the list
            bp_volume,myo_volume = cal_bp_volume(file_path)
            df.loc[len(df)] = [int(time_step),bp_volume*1e6,df_PINN_400['pressure_LV'][int(time_step)*2]]

    df_sorted = df.sort_values(by='time_step')
    print(df_sorted)
    df_sorted.to_csv(f'{directory_path}/P_volumes_meshcode.csv', index=False)

def calculate_volumes_comsol(directory_path):

    # List to store outputs

    df = pd.DataFrame(columns=['time_step', 'bp_volume', 'pressure'])

    # Loop through all files in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file has a .vtk extension
        if file_name.endswith('.vtu'):
            print(file_name)
            time_step = file_name.split('_')[1].split('.')[0][1:]
            print(time_step)
            # Full file path
            file_path = os.path.join(directory_path, file_name)

            # Process the file and store the output in the list
            bp_volume,myo_volume = cal_bp_volume(file_path)
            df.loc[len(df)] = [int(time_step),bp_volume*1e6,int(time_step)]

    df_sorted = df.sort_values(by='time_step')
    print(df_sorted)
    df_sorted.to_csv(f'{directory_path}/P_volumes.csv', index=False)


#directory_path = "/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/COMSOL/EDP_10/EDP_10_template"
directory_path = "/data.lfpn/ibraun/Code/Cardio-PINN/Synthetic_shapes/LV_mean_human/EDP_15/Epochs_500"
calculate_volumes_PINN(directory_path)