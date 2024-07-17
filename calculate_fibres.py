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