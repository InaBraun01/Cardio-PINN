import vtk
import os
import sys
import csv
import pandas as pd
import numpy as np

# Step 2: Load the VTK file
def load_vtk_unstructured_grid(filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# Step 3: Convert to PolyData (this does not change the geometry of the anatomy) (This is a surface mesh)
def convert_to_polydata(unstructured_grid):
    surface_filter = vtk.vtkGeometryFilter()
    surface_filter.SetInputData(unstructured_grid)
    surface_filter.Update()
    return surface_filter.GetOutput()

# Step 4: Calculate the Volume (Can only be done on a PolyData object)
def calculate_volume(polydata):
    # Create a mass properties object to compute volume
    mass_props = vtk.vtkMassProperties()
    mass_props.SetInputData(polydata)
    volume = mass_props.GetVolume()
    return volume

# Step 3: Cap the open part of the bowl to make it a closed surface
def cap_bowl(polydata):
    # Create a triangulated surface to close the open end of the bowl
    boundary_edges = vtk.vtkFeatureEdges()
    boundary_edges.SetInputData(polydata)
    boundary_edges.BoundaryEdgesOn()
    boundary_edges.FeatureEdgesOff()
    boundary_edges.NonManifoldEdgesOff()
    boundary_edges.ManifoldEdgesOff()
    boundary_edges.Update()

    boundary_polydata = vtk.vtkPolyData()
    boundary_polydata.SetPoints(boundary_edges.GetOutput().GetPoints())
    boundary_polydata.SetLines(boundary_edges.GetOutput().GetLines())

    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(boundary_polydata)
    delaunay.Update()

    # Append the original polydata with the cap
    append_filter = vtk.vtkAppendPolyData()
    append_filter.AddInputData(polydata)
    append_filter.AddInputData(delaunay.GetOutput())
    append_filter.Update()

    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(append_filter.GetOutput())
    clean_filter.Update()

    return clean_filter.GetOutput()

# Step 4: Create a solid mesh from the closed surface
def create_solid_mesh(closed_surface):
    delaunay3D = vtk.vtkDelaunay3D()
    delaunay3D.SetInputData(closed_surface)
    delaunay3D.Update()

    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(delaunay3D.GetOutputPort())
    surface_filter.Update()

    return surface_filter.GetOutput()

# Step 5: Calculate the volume using vtkMassProperties
def calculate_volume(polydata):
    mass_props = vtk.vtkMassProperties()
    mass_props.SetInputData(polydata)
    volume = mass_props.GetVolume()
    return volume


