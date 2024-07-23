
# import os
# import pyvista as pv
# import numpy as np


# # Read the original model from a VTK file
# mesh = pv.read('combined_mesh.vtk')

# # Define the scaling factor to change the unit from m to mm
# scale_factor = 1/1000  #go back to unit of mm

# # Scale the mesh
# scaled_mesh = mesh.scale([scale_factor, scale_factor, scale_factor])

# # Save the sampled grid to a new VTK file
# scaled_mesh.save('Scaled_combined_mesh.vtk')

import os
import pyvista as pv
import numpy as np

# Read the original model from a VTK file
mesh = pv.read('LV_mean_half_scaled.vtk')

# Define the scaling factor to change the unit from m to mm
scale_factor = 1/1000  # go back to unit of mm

# Scale the mesh
scaled_mesh = mesh.scale([scale_factor, scale_factor, scale_factor], inplace=False)

# Copy all point data (features) from the original mesh to the scaled mesh
for key in mesh.point_data.keys():
    scaled_mesh.point_data[key] = mesh.point_data[key]

# Save the scaled mesh with all features to a new VTK file
scaled_mesh.save('LV_mean_correct_scaled.vtk')

print("Scaling complete")
