
import sys
import numpy as np
import os
import pyvista as pv

def Scale_mesh(vtk_file, data_file, out_folder):
    """In this function the input mesh in which the mesh position is given in the unit mm is scaled,
    so that the positons of the ndoes are given in m """

    # Read the original model from a VTK file
    mesh = pv.read(data_file + vtk_file)

    # Define the scaling factor to change the unit from m to mm
    scale_factor = 1/1000  # go back to unit of mm

    # Scale the mesh
    scaled_mesh = mesh.scale([scale_factor, scale_factor, scale_factor], inplace=False)

    # Copy all point data (features) from the original mesh to the scaled mesh
    for key in mesh.point_data.keys():
        scaled_mesh.point_data[key] = mesh.point_data[key]

    # Save the scaled mesh with all features to a new VTK file
    scaled_mesh.save(f"{out_folder}/scaled_{vtk_file}")

    print("Scaling complete")
    return

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

    #load in scalar values at each node
    labels  = vtk_to_numpy(data.GetPointData().GetArray('labels'))
    x_c  = vtk_to_numpy(data.GetPointData().GetArray('x_c'))
    x_l  = vtk_to_numpy(data.GetPointData().GetArray('x_l'))
    x_t  = vtk_to_numpy(data.GetPointData().GetArray('x_t'))
    #load in the vectors at each node
    # f_vector  = vtk_to_numpy(data.GetPointData().GetVectors('f'))
    # s_vector  = vtk_to_numpy(data.GetPointData().GetVectors('s'))

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


def calculate_f(n, x_t, T_epi, T_endo):
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

def GenerateFibres(fibres_filename,Fiber_params):

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    #Read the VTK file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(fibres_filename)
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

    print(scalarArray)
    numPoints = gradientArray.GetNumberOfTuples() #calculate number of nodes (number of points for which gradient is calculated)

    f_x = np.zeros((numPoints,1))
    f_y = np.zeros((numPoints,1))
    f_z = np.zeros((numPoints,1))

    s_x = np.zeros((numPoints,1))
    s_y = np.zeros((numPoints,1))
    s_z = np.zeros((numPoints,1))

    for i in range(numPoints): #loop through all of the nodes
        n = gradientArray.GetTuple3(i) #get the tuple out of the gradient_array for that node
        n_normalized = normalize_vector(n) #normaluse the vector
        x_t = scalarArray.GetValue(i) #get the value of x_t at that node position
        
        f = calculate_f(n_normalized, x_t,Fiber_params.epi_fiber_angle,Fiber_params.endo_fiber_angle) #calculate the fibre direction at each node
        
        f_normalized = normalize_vector(f) #normalise the fibre direction
        f_x[i] = f_normalized[0]
        f_y[i] = f_normalized[1]
        f_z[i] = f_normalized[2]

        s_normalized = calculate_orthogonal_vector(n_normalized,f_normalized)
        s_x[i] = s_normalized[0]
        s_y[i] = s_normalized[1]
        s_z[i] = s_normalized[2]

    return f_x.squeeze(), f_y.squeeze(), f_z.squeeze(),s_x.squeeze(), s_y.squeeze(), s_z.squeeze()

