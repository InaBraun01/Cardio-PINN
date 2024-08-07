import sys
import vtk
import numpy as np
from volume_cal_functions import load_vtk_unstructured_grid, convert_to_polydata, calculate_volume, cap_bowl, create_solid_mesh, calculate_volume
import monkey_functions as monkey
from collections import defaultdict

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
    Coords, Els, n_points,n_el, Node_par_coords ,Faces_Endo = monkey.LoadModelAnatomy(vtk_file_path)
    
    volume_blood_ML = 0
    for j in range(len(Faces_Endo)): #for each face of the tetrahedral cells
        sel_el = Faces_Endo[j] #for all nodes on the face
        oa = np.array(Coords[sel_el[0],:]) #x,y,z coordinates of the node
        ob = np.array(Coords[sel_el[1],:]) #x,y,z coordinates of the node
        oc = np.array(Coords[sel_el[2],:]) #x,y,z coordinates of the node
        volume_blood_ML += 1.0/6.0*abs(np.dot(np.cross(oa,ob),oc))#*1e6  #add together individual volumn parts

    return volume_blood_ML


#vtk_file_path = "Synthetic_shapes/Maike_ED/PINN_data/scaled_Maike_ED.vtk"
vtk_file_path = "Test_data/Maike_ED.vtk"
#vtk_file_path = "Test_data/LV_mean_half_scaled.vtk"

mesh_bp_vol, mesh_myo_vol = cal_bp_volume(vtk_file_path)
PINN_bp_vol = Compute_Volume(vtk_file_path)

print(f"The calculated Blood pool volume from the meshes is:{mesh_bp_vol}")
print(f"The calculated Myocardium volume from the meshes is:{mesh_myo_vol}")
print(f"The calculated blood pool volume from the PINN is: {PINN_bp_vol}")

# Load the VTK file
unstructured_grid = load_vtk_unstructured_grid(vtk_file_path)

# Extract the surface from the unstructured grid (convert to polydata)
polydata = convert_to_polydata(unstructured_grid)

writer = vtk.vtkPolyDataWriter()
writer.SetFileName('mesh_output.vtk')
writer.SetInputData(polydata)
writer.Write()


# # #Check if mesh calculation is sensible
# Coords, Els, n_points,n_el, Node_par_coords ,Faces_Endo = monkey.LoadModelAnatomy(vtk_file_path)
# # # Count edges shared by more than two faces
# # edge_count = defaultdict(int)
# # for face in Faces_Endo:
# #     for i in range(3):
# #         edge = tuple(sorted([face[i], face[(i+1)%3]]))
# #         edge_count[edge] += 1
# # problematic_edges = [e for e, count in edge_count.items() if count != 2]
# # print(f"Edges not shared by exactly 2 faces: {len(problematic_edges)}")


# def face_normal(face):
#     a, b, c = [np.array(Coords[i]) for i in face]
#     return np.cross(b-a, c-a)

# normals = [face_normal(face) for face in Faces_Endo]
# consistent = all(np.dot(normals[0], n) > 0 for n in normals)
# print(f"Face orientations consistent: {consistent}")


# # centroid = np.mean(Coords, axis=0)
# # print(f"Mesh centroid: {centroid}")
# # # Ensure this point is inside your bowl

# # def tetrahedron_volume(a, b, c, ref_point):
# #     return np.dot(np.cross(a - ref_point, b - ref_point), c - ref_point) / 6.0

# # centroid = np.mean(Coords, axis=0)
# # volume = sum(tetrahedron_volume(Coords[face[0]], Coords[face[1]], Coords[face[2]], centroid) 
# #              for face in Faces_Endo)
# # print(f"Volume (surface integral): {abs(volume)}")

# def fix_orientations(faces, coords):
#     # Compute a reference normal
#     ref_normal = face_normal(faces[0])
    
#     # Fix orientations
#     for i, face in enumerate(faces):
#         normal = face_normal(face)
#         if np.dot(normal, ref_normal) < 0:
#             faces[i] = face[::-1]  # Reverse the face
    
#     return faces

# Faces_Endo_new = fix_orientations(Faces_Endo, Coords)

# normals = [face_normal(face) for face in Faces_Endo]
# consistent = all(np.dot(normals[0], n) > 0 for n in normals)
# print(f"Face orientations consistent: {consistent}")

# def calc_volume(faces, coords):
#     centroid = np.mean(coords, axis=0)
#     volume = 0
#     for face in faces:
#         a, b, c = [np.array(coords[i]) for i in face]
#         volume += np.dot(np.cross(a - centroid, b - centroid), c - centroid) / 6.0
#     return abs(volume)

# volume = calc_volume(Faces_Endo_new, Coords)
# print(f"Calculated volume: {volume}")