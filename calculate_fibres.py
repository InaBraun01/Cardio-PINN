#use meshes for which n is already calculated for every node
#use n to calculate f at every node


import vtk
import numpy as np

# Constants (replace with your actual values)
T_epi = -np.pi/3  # Example value, replace with your actual value
T_endo = np.pi/3  # Example value, replace with your actual value

def calculate_f(n, x_t):
    nx, ny, nz = n
    csA = np.cos(T_epi * x_t + T_endo * (1 - x_t))
    snA = np.sin(T_epi * x_t + T_endo * (1 - x_t))
    
    f1 = (csA + nx**2 * (1-csA)) * ny + (nx*ny*(1-csA) - nz*snA) * (-nx)
    f2 = (ny*nx*(1-csA) + nz*snA) * ny + (csA + ny**2 * (1-csA)) * (-nx)
    f3 = (nz*nx*(1-csA) - ny*snA) * ny + (nz*ny*(1-csA) + nx*snA) * (-nx)
    
    return (f1, f2, f3)

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return tuple(x / norm for x in v) if norm != 0 else (0, 0, 0)

# Read the VTK file
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("Scaled_Mesh_Maike.vtk")
reader.Update()

# Get the data
data = reader.GetOutput()

# Ensure the scalar data is active
data.GetPointData().SetActiveScalars("x_t")

# Create a gradient filter
gradientFilter = vtk.vtkGradientFilter()
gradientFilter.SetInputData(data)
gradientFilter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "x_t")
gradientFilter.SetResultArrayName("x_t_gradient")
gradientFilter.SetComputeGradient(True)
gradientFilter.Update()

# Get the output
gradientData = gradientFilter.GetOutput()
gradientArray = gradientData.GetPointData().GetArray("x_t_gradient")
scalarArray = data.GetPointData().GetArray("x_t")

# Calculate the normalized gradient and f vector
numPoints = gradientArray.GetNumberOfTuples()
normalizedGradientArray = vtk.vtkDoubleArray()
normalizedGradientArray.SetNumberOfComponents(3)
normalizedGradientArray.SetNumberOfTuples(numPoints)
normalizedGradientArray.SetName("n")

fArray = vtk.vtkDoubleArray()
fArray.SetNumberOfComponents(3)
fArray.SetNumberOfTuples(numPoints)
fArray.SetName("f_vector")

normalizedFArray = vtk.vtkDoubleArray()
normalizedFArray.SetNumberOfComponents(3)
normalizedFArray.SetNumberOfTuples(numPoints)
normalizedFArray.SetName("f")

for i in range(numPoints):
    n = gradientArray.GetTuple3(i)
    n_normalized = normalize_vector(n)
    x_t = scalarArray.GetValue(i)
    
    normalizedGradientArray.SetTuple3(i, *n_normalized)
    
    f = calculate_f(n_normalized, x_t)
    fArray.SetTuple3(i, *f)
    
    f_normalized = normalize_vector(f)
    normalizedFArray.SetTuple3(i, *f_normalized)

# Create a new vtkUnstructuredGrid with all original data and new vectors
new_data = vtk.vtkUnstructuredGrid()
new_data.DeepCopy(data)  # This copies all the original data

# Add the new arrays
new_data.GetPointData().AddArray(normalizedGradientArray)
new_data.GetPointData().AddArray(normalizedFArray)


# Write the result to a new VTK file
writer = vtk.vtkUnstructuredGridWriter()
writer.SetInputData(new_data)
writer.SetFileName("output_with_normalized_vectors.vtk")
writer.Write()