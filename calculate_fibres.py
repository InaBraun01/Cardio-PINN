# import vtk
# import sys


# # Create a reader for the VTK file
# reader = vtk.vtkUnstructuredGridReader()
# reader.SetFileName('Scaled_Mesh_Maike.vtk')  # Replace with your file path
# reader.Update()

# # Get the output unstructured grid
# mesh = reader.GetOutput()

# # Access the scalar field 'x_t' (assuming it's a point data array)
# x_t_array = mesh.GetPointData().GetArray('x_t')  # Replace 'x_t' with your actual scalar field name

# # Create a vtkGradientFilter to compute gradients
# gradient_filter = vtk.vtkGradientFilter()
# gradient_filter.SetInputData(mesh)
# gradient_filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'x_t')
# gradient_filter.Update()

# # Get the gradient array
# gradient_array = gradient_filter.GetOutput().GetPointData().GetArray('Gradient')

# # Example: Accessing gradient values at a specific point (e.g., first point)
# gradient_values = gradient_array.GetTuple3(0)  # Replace 0 with the index of the point you're interested in

# # Print the gradient values
# print("Gradient at point 0:", gradient_values)


import vtk

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

# Get the output and add it to the original data
gradientData = gradientFilter.GetOutput()
gradientArray = gradientData.GetPointData().GetArray("x_t_gradient")
data.GetPointData().AddArray(gradientArray)

# Write the result to a new VTK file
writer = vtk.vtkUnstructuredGridWriter()
writer.SetInputData(data)
writer.SetFileName("output_test.vtk")
writer.Write()