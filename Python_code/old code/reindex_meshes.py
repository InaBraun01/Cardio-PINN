import vtk
import numpy as np
from scipy.spatial import cKDTree

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

def get_points_and_cells(grid):
    points = np.array([grid.GetPoint(i) for i in range(grid.GetNumberOfPoints())])
    cells = []
    for i in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(i)
        cell_points = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        cells.append((cell.GetCellType(), cell_points))
    return points, cells

def reindex_points(points1, points2, tolerance=1e-6):
    # Combine all unique points
    all_points = np.vstack((points1, points2))
    unique_points, inverse = np.unique(all_points.round(decimals=int(-np.log10(tolerance))), axis=0, return_inverse=True)
    
    # Create new indices
    new_indices1 = inverse[:len(points1)]
    new_indices2 = inverse[len(points1):]
    
    return unique_points, new_indices1, new_indices2

def update_cells(cells, new_indices):
    new_cells = []
    for cell_type, cell_points in cells:
        new_cell_points = [new_indices[p] for p in cell_points]
        new_cells.append((cell_type, new_cell_points))
    return new_cells

# Read the two VTK files
grid1 = read_vtk('Mesh_Maike.vtk')
grid2 = read_vtu('fibres_Maike.vtu')

# Extract points and cells from both grids
points1, cells1 = get_points_and_cells(grid1)
points2, cells2 = get_points_and_cells(grid2)

# Reindex points
unique_points, new_indices1, new_indices2 = reindex_points(points1, points2)

# Update cells with new point indices
new_cells1 = update_cells(cells1, new_indices1)
new_cells2 = update_cells(cells2, new_indices2)

# Create new grids with reindexed points and cells
def create_new_grid(points, cells):
    new_grid = vtk.vtkUnstructuredGrid()
    new_points = vtk.vtkPoints()
    for point in points:
        new_points.InsertNextPoint(point)
    new_grid.SetPoints(new_points)
    
    for cell_type, cell_points in cells:
        id_list = vtk.vtkIdList()
        for p in cell_points:
            id_list.InsertNextId(p)
        new_grid.InsertNextCell(cell_type, id_list)
    return new_grid

new_grid1 = create_new_grid(unique_points, new_cells1)
new_grid2 = create_new_grid(unique_points, new_cells2)

# Write the reindexed grids to new VTK files
write_vtk(new_grid1, 'reindexed_mesh_maike.vtk')
write_vtk(new_grid2, 'reindexed_fibres_maike.vtk')

print("Reindexing complete. New meshes saved as 'reindexed_mesh1.vtk' and 'reindexed_mesh2.vtk'")