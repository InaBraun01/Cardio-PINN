#use meshes for which n is already calculated for every node
#use n to calculate f at every node


import vtk
import numpy as np
import sys

# Constants (replace with your actual values)
T_epi = -np.pi/3  # Example value, replace with your actual value
T_endo = np.pi/3  # Example value, replace with your actual value

def calculate_f(n, x_t):
    nx, ny, nz = n #split tuple n into three components
    csA = np.cos(T_epi * x_t + T_endo * (1 - x_t))
    snA = np.sin(T_epi * x_t + T_endo * (1 - x_t))
    #calculate individual components of fibre direction f
    f1 = (csA + nx**2 * (1-csA)) * ny + (nx*ny*(1-csA) - nz*snA) * (-nx) 
    f2 = (ny*nx*(1-csA) + nz*snA) * ny + (csA + ny**2 * (1-csA)) * (-nx)
    f3 = (nz*nx*(1-csA) - ny*snA) * ny + (nz*ny*(1-csA) + nx*snA) * (-nx)
    #return tuple with vector describing fibre direction
    return (f1, f2, f3)

def normalize_vector(v):
    #normalise vector v which is saved in form of a tuple with three components
    norm = np.linalg.norm(v)
    return tuple(x / norm for x in v) if norm != 0 else (0, 0, 0)

def calculate_orthogonal_vector(v1, v2):
    # Calculates a unit vector which is orthogonal to the two input vectors
    cross = np.cross(v1, v2)
    return normalize_vector(cross)

# Read the VTK file
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("Scaled_Mesh_Maike.vtk")
reader.Update()  #exectues the reader pipline up until here and thus actually reads in the file

# Get the data from the vtk file
data = reader.GetOutput() 

# get x_t value for every node
data.GetPointData().SetActiveScalars("x_t")

#Calculate the gradient over the mesh with respect to x_t
gradientFilter = vtk.vtkGradientFilter()
gradientFilter.SetInputData(data)
gradientFilter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "x_t") #Look for the scalar data named 'x_t' in the point data of the input dataset
# vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS describes where the data is located:
# FIELD_ASSOCIATION_POINTS: data is associated with the individual nodes and not for example with cells or the entire data set
gradientFilter.SetResultArrayName("x_t_gradient")
gradientFilter.SetComputeGradient(True)
gradientFilter.Update() # actually calculate the gradient value at every node

# Get the output
gradientData = gradientFilter.GetOutput()  # get output from the gradient filter
gradientArray = gradientData.GetPointData().GetArray("x_t_gradient") #for every node extract the calculated gradient
scalarArray = data.GetPointData().GetArray("x_t") #for every node extract the value of x_t

# Calculate the normalized gradient and f vector
numPoints = gradientArray.GetNumberOfTuples() #calculate number of nodes (number of points for which gradient is calculated)
normalizedGradientArray = vtk.vtkDoubleArray()  #initialize vtk array into which the normalized gradient vectors can be saved
normalizedGradientArray.SetNumberOfComponents(3) # each saved vector should have 3 components 
normalizedGradientArray.SetNumberOfTuples(numPoints) #a vector should be saved for each node point
normalizedGradientArray.SetName("n") #name for new vector saved at every node

fArray = vtk.vtkDoubleArray() #initialize vtk array into which the calculated fibre direction should be saved
fArray.SetNumberOfComponents(3)
fArray.SetNumberOfTuples(numPoints)
fArray.SetName("f_vector")

normalizedFArray = vtk.vtkDoubleArray() #initialize vtk array into which the calculated normalised fibre direction should be saved
normalizedFArray.SetNumberOfComponents(3)
normalizedFArray.SetNumberOfTuples(numPoints)
normalizedFArray.SetName("f")

normalizedSArray = vtk.vtkDoubleArray() #initialize vtk array into which the calculated normalised sheet direction is saved
normalizedSArray.SetNumberOfComponents(3)
normalizedSArray.SetNumberOfTuples(numPoints)
normalizedSArray.SetName("s")

for i in range(numPoints): #loop through all of the nodes
    n = gradientArray.GetTuple3(i) #get the tuple out of the gradient_array for that node
    n_normalized = normalize_vector(n) #normaluse the vector
    x_t = scalarArray.GetValue(i) #get the value of x_t at that node position
    
    normalizedGradientArray.SetTuple3(i, *n_normalized) # add the normalised gradient vector to the vtk vector
    
    f = calculate_f(n_normalized, x_t) #calculate the fibre direction at each node
    fArray.SetTuple3(i, *f) # add the calculated fibre direction to the vtk vector
    
    f_normalized = normalize_vector(f) #normalise the fibre direction
    normalizedFArray.SetTuple3(i, *f_normalized) #add the normalised fibre direction to the vtk vector

    s_normalized = calculate_orthogonal_vector(n_normalized,f_normalized)
    normalizedSArray.SetTuple3(i,*s_normalized)

# Create a new vtkUnstructuredGrid with all original data and new vectors
new_data = vtk.vtkUnstructuredGrid()
new_data.DeepCopy(data)  # This copies all the original data

# Add the new arrays (only want to save the normalised vectors)
new_data.GetPointData().AddArray(normalizedGradientArray)
new_data.GetPointData().AddArray(normalizedFArray)
new_data.GetPointData().AddArray(normalizedSArray)


# Write the result to a new VTK file
writer = vtk.vtkUnstructuredGridWriter()
writer.SetInputData(new_data)
writer.SetFileName("Scaled_Maike_n_s_f.vtk")
writer.Write()