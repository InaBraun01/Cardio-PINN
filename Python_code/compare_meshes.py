import vtk
import numpy as np

# read in .vtk file
def read_vtk_unstructured_grid(file_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

# read in .vtu file
def read_vtu_unstructured_grid(file_path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()  
    return reader.GetOutput()

# File paths
file_path1 = 'Mesh_Maike.vtk'
file_path2 = 'fibres_Maike.vtu'

# Read the VTK files
mesh1 = read_vtk_unstructured_grid(file_path1)
mesh2 = read_vtu_unstructured_grid(file_path2)

# Print out coordinates of points at a specific index
index_list = [0,5,10,50,100,200,400,800]

points1 = mesh1.GetPoints() #retrieves all of the points
points2 = mesh2.GetPoints()

for index in index_list:
    #point at index
    point1 = points1.GetPoint(index)
    point2 = points2.GetPoint(index)
    print(f"Coordinate of points of initial mesh {point1}")
    print(f"Coordinate of points of mesh woth fibres {point2}\n")



# # Step 2: Compute the average and RMS point-to-point distances
# average_distance, rms_distance = compute_point_to_point_distances(mesh1, mesh2)
# print(f"Average Point-to-Point Distance: {average_distance}")
# print(f"RMS Point-to-Point Distance: {rms_distance}")

#print some points at certain position randomly
#see if they have the same location => the points are not at the same location
#then just sort them by position