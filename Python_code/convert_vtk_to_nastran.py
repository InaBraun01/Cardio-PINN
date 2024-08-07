import vtk
import numpy as np

# Read the VTK file
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("Synthetic_shapes/Shape1/Anatomy.vtk")
reader.Update()

# Get the mesh data
mesh = reader.GetOutput()

# Extract points and cells
points = mesh.GetPoints()
cells = mesh.GetCells()

# Write to .nas or .bdf format
with open("Synthetic_shapes/Shape1/Anatomy.nas", "w") as f:
    # Write header
    f.write("$ Nastran input file\n")
    
    # Write node data
    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        f.write(f"GRID,{i+1},,{point[0]},{point[1]},{point[2]}\n")
    
    # Write element data
    # This part will depend on your specific element types
    # You'll need to map VTK cell types to Nastran element types
    
    # Example for tetrahedral elements:
    for i in range(cells.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        if cell.GetCellType() == vtk.VTK_TETRA:
            pts = cell.GetPointIds()
            f.write(f"CTETRA,{i+1},1,{pts.GetId(0)+1},{pts.GetId(1)+1},{pts.GetId(2)+1},{pts.GetId(3)+1}\n")

# Note: This is a simplified example and may need to be adapted based on your specific mesh structure

