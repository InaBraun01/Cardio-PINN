import vtk
import numpy as np

def read_vtk(filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# read in .vtu file
def read_vtu(file_path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()  
    return reader.GetOutput()

def write_vtk(grid, filename):
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

def get_point_data(grid):
    point_data = {}
    for i in range(grid.GetPointData().GetNumberOfArrays()):
        array = grid.GetPointData().GetArray(i)
        name = array.GetName()
        data = np.array([array.GetTuple(j) for j in range(array.GetNumberOfTuples())])
        point_data[name] = data
    return point_data

def match_exact_points(points1, points2):
    # Create a dictionary of points2 for fast lookup
    points2_dict = {tuple(p): i for i, p in enumerate(points2)}
    
    # Match points1 to points2
    indices = np.array([points2_dict.get(tuple(p), -1) for p in points1])
    
    return indices

# Read the two VTK files
grid1 = read_vtk('Mesh_Maike.vtk')
grid2 = read_vtu('fibres_Maike.vtu')
# Get points from both grids
points1 = np.array([grid1.GetPoint(i) for i in range(grid1.GetNumberOfPoints())])
points2 = np.array([grid2.GetPoint(i) for i in range(grid2.GetNumberOfPoints())])

# Match points between the two grids
match_indices = match_exact_points(points1, points2)

# Check if all points were matched
if np.any(match_indices == -1):
    print("Warning: Some points in mesh1 do not have exact matches in mesh2")

# Get point data from both grids
point_data1 = get_point_data(grid1)
point_data2 = get_point_data(grid2)

# Create a new grid with the geometry from grid1
new_grid = vtk.vtkUnstructuredGrid()
new_grid.DeepCopy(grid1)

# Combine and add point data to the new grid
for name, data in point_data1.items():
    array = vtk.vtkDoubleArray()
    array.SetName(name)
    array.SetNumberOfComponents(data.shape[1] if len(data.shape) > 1 else 1)
    array.SetNumberOfTuples(data.shape[0])
    for i, value in enumerate(data):
        array.SetTuple(i, value if isinstance(value, (list, np.ndarray)) else [value])
    new_grid.GetPointData().AddArray(array)

for name, data in point_data2.items():
    if name in point_data1:
        name = f"{name}_2"  # Avoid name conflicts
    array = vtk.vtkDoubleArray()
    array.SetName(name)
    array.SetNumberOfComponents(data.shape[1] if len(data.shape) > 1 else 1)
    array.SetNumberOfTuples(len(points1))
    for i, match_idx in enumerate(match_indices):
        if match_idx != -1:
            value = data[match_idx]
            array.SetTuple(i, value if isinstance(value, (list, np.ndarray)) else [value])
        else:
            array.SetTuple(i, [0] * array.GetNumberOfComponents())  # Fill with zeros if no match
    new_grid.GetPointData().AddArray(array)

# Write the combined grid to a new VTK file
write_vtk(new_grid, 'combined_mesh.vtk')

print("Combination complete. New mesh saved as 'combined_mesh.vtk'")