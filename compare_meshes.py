import vtk
import numpy as np

def read_vtk_unstructured_grid(file_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def compute_point_to_point_distances(mesh1, mesh2):
    points1 = mesh1.GetPoints()
    points2 = mesh2.GetPoints()
    
    if points1.GetNumberOfPoints() != points2.GetNumberOfPoints():
        raise ValueError("Meshes have different numbers of points.")
    
    distances = []
    for i in range(points1.GetNumberOfPoints()):
        p1 = np.array(points1.GetPoint(i))
        p2 = np.array(points2.GetPoint(i))
        distance = np.linalg.norm(p1 - p2)
        distances.append(distance)
    
    distances = np.array(distances)
    return distances.mean(), np.sqrt(np.mean(distances**2))

# File paths
file_path1 = 'Synthetic_shapes/Bobo_fit_5_modes/scaled_Anatomy.vtk'
file_path2 = 'Synthetic_shapes/Bobo_fit_5_modes/PINN_data_EDP_10/Simulation_results/Displ_40.vtk'

# Step 1: Read the VTK files
mesh1 = read_vtk_unstructured_grid(file_path1)
mesh2 = read_vtk_unstructured_grid(file_path2)

# Step 2: Compute the average and RMS point-to-point distances
average_distance, rms_distance = compute_point_to_point_distances(mesh1, mesh2)
print(f"Average Point-to-Point Distance: {average_distance}")
print(f"RMS Point-to-Point Distance: {rms_distance}")
