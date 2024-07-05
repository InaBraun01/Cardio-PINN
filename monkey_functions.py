
import sys
def LoadModelAnatomy(vtk_mean):

    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    #read in vtk structure from input file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_mean)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    data = reader.GetOutput()

    n_points = data.GetNumberOfPoints()   #number of points
    n_el     = data.GetNumberOfCells()    #number of cells
    Coords   =  vtk_to_numpy(data.GetPoints().GetData())   #coordinates of each point
    Els      = np.zeros((n_el,4),dtype=int)  #connections
    for i in range(n_el):
        cell_type = data.GetCellType(i)
        n_nodes_el   = data.GetCell(i).GetPointIds().GetNumberOfIds()    #number of vertices for each cell
        for n_sel in range(n_nodes_el):
            Els[i,n_sel] = int(data.GetCell(i).GetPointId(n_sel)) # save list of vertices contained in cell

    labels  = vtk_to_numpy(data.GetPointData().GetArray('labels'))
    x_c  = vtk_to_numpy(data.GetPointData().GetArray('x_c'))
    x_l  = vtk_to_numpy(data.GetPointData().GetArray('x_l'))
    x_t  = vtk_to_numpy(data.GetPointData().GetArray('x_t'))
    # e_c  = vtk_to_numpy(data.GetPointData().GetVectors('e_c'))
    # e_l  = vtk_to_numpy(data.GetPointData().GetVectors('e_l'))
    # e_t  = vtk_to_numpy(data.GetPointData().GetVectors('e_t'))

    Node_par_coords = np.zeros((n_points,4))
    Node_par_coords[:,0] = labels
    Node_par_coords[:,1] = x_c
    Node_par_coords[:,2] = x_l
    Node_par_coords[:,3] = x_t

    faces_connectivity = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])  #vertices connected on the 4 faces of the tetrahedron
    Faces_Endo = []  #list of points that make up tetrahedral cell with the one specific vertex
    start_faces = True
    for kk in range(n_el): #for each cell 
        el_points = Els[kk,:] #vertices found in kk cell, there are always 4 vertices in each cell
        for jj in range(4): #for all vertices making up the cell, save the connection of vertices on the faces
            if all(labels[int(v)] == 2 for v in el_points[faces_connectivity[jj]]):
                if start_faces:
                    Faces_Endo  = np.array(el_points[faces_connectivity[jj]],dtype=int).reshape(1,-1)
                    start_faces = False
                else:
                    Faces_Endo = np.concatenate((Faces_Endo,np.array(el_points[faces_connectivity[jj]],dtype=int).reshape(1,-1)),0)

    #return Coords, Els, n_points, n_el, Node_par_coords, e_t, e_l, e_c, Faces_Endo
    return Coords, Els, n_points, n_el, Node_par_coords, Faces_Endo


def GenerateFibres(fibres_filename):

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    # Read the VTU file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(fibres_filename) #for each node have all 3 components of the fibre, sheet, normal direction stored
    reader.Update()

    # Get the unstructured grid
    grid = reader.GetOutput()

    # Get information about point data
    point_data = grid.GetPointData()
    num_arrays = point_data.GetNumberOfArrays()


    f_x  = vtk_to_numpy(grid.GetPointData().GetArray('First_basis_vector,_X-component'))
    f_y  = vtk_to_numpy(grid.GetPointData().GetArray('First_basis_vector,_Y-component'))
    f_z  = vtk_to_numpy(grid.GetPointData().GetArray('First_basis_vector,_Z-component'))

    s_x  = vtk_to_numpy(grid.GetPointData().GetArray('Second_basis_vector,_X-component'))
    s_y  = vtk_to_numpy(grid.GetPointData().GetArray('Second_basis_vector,_Y-component'))
    s_z  = vtk_to_numpy(grid.GetPointData().GetArray('Second_basis_vector,_Z-component'))

    return f_x,f_y,f_z,s_x,s_y,s_z

f_x,f_y,f_z,s_x,s_y,s_z = GenerateFibres("fibres_Maike.vtu")

print(f_x)
print(f_y)
print(f_z)
print(s_x)
print(s_y)
print(s_z)
