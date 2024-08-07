import numpy as np
import vtk
from volume_cal_functions import load_vtk_unstructured_grid, convert_to_polydata, calculate_volume, cap_bowl, create_solid_mesh, calculate_volume
import monkey_functions as monkey

def initial_Compute_Volume(vtk_file_path):
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
    return bp_volume


def Compute_Volume_my(vtk_file):

    ''' Compute left ventricular blood pool volume'''
    Coords, Els, n_points,n_el, Node_par_coords ,Faces_Endo = monkey.LoadModelAnatomy(vtk_file)

    points = vtk.vtkPoints()
    for coord in Coords:
        points.InsertNextPoint(coord)

    # Create triangles
    triangles = vtk.vtkCellArray()
    for face in Faces_Endo:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, int(face[0]))
        triangle.GetPointIds().SetId(1, int(face[1]))
        triangle.GetPointIds().SetId(2, int(face[2]))
        triangles.InsertNextCell(triangle)

    # Create a polydata object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName('mesh_output_test.vtk')
    writer.SetInputData(polydata)
    writer.Write()

    mass_properties = vtk.vtkMassProperties()
    mass_properties.SetInputData(polydata)
    volume = mass_properties.GetVolume()

    volume_blood_ML = 0
    for j in range(len(Faces_Endo)): #for each face of the tetrahedral cells
        sel_el = Faces_Endo[j] #for all nodes on the face
        oa = np.array(Coords[sel_el[0],:]) #x,y,z coordinates of the node
        ob = np.array(Coords[sel_el[1],:]) #x,y,z coordinates of the node
        oc = np.array(Coords[sel_el[2],:]) #x,y,z coordinates of the node
        volume_blood_ML += 1.0/6.0*abs(np.dot(np.cross(oa,ob),oc))  #add together individual volumn parts

    print(f"The volume of the mesh is: {volume} and {volume_blood_ML}")

    return volume

# def Compute_Volume_their(vtk_file):

#     Coords, Els, n_points,n_el, Node_par_coords ,Faces_Endo = monkey.LoadModelAnatomy(vtk_file)
    
#     volume_blood_ML = 0
#     for j in range(len(Faces_Endo)): #for each face of the tetrahedral cells
#         sel_el = Faces_Endo[j] #for all nodes on the face
#         oa = np.array(Coords[sel_el[0],:]) #x,y,z coordinates of the node
#         ob = np.array(Coords[sel_el[1],:]) #x,y,z coordinates of the node
#         oc = np.array(Coords[sel_el[2],:]) #x,y,z coordinates of the node
#         volume_blood_ML += 1.0/6.0*abs(np.dot(np.cross(oa,ob),oc))  #add together individual volumn parts

#         grid = pv.UnstructuredGrid(Coords, Faces_Endo)
#         grid.save("unstructured_grid.vtk")

#     print(f"The volume of the mesh is: {volume_blood_ML}")
#     return volume_blood_ML


vtk_file_path = "../Test_data/Maike_ED.vtk"
Compute_Volume_my(vtk_file_path)
#Compute_Volume_their(vtk_file_path)