
import os
import pyvista as pv
import numpy as np


# Read the original model from a VTK file
mesh = pv.read('Synthetic_shapes/Bobo_fit_5_modes/Anatomy.vtk')

# Define the scaling factor to change the unit from m to mm
scale_factor = 1/1000  #go back to unit of mm

# Scale the mesh
scaled_mesh = mesh.scale([scale_factor, scale_factor, scale_factor])

# Save the sampled grid to a new VTK file
scaled_mesh.save('Synthetic_shapes/Bobo_fit_5_modes/scaled_Anatomy.vtk')

