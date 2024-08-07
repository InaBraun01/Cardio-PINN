import sys,os,shutil
import tensorflow.compat.v1 as tf #code was written for the older tensorflow version 1.10
tf.disable_v2_behavior()  #disable version 2 behavior of tensor flow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import math 

import vtk
from   vtk.util.numpy_support import vtk_to_numpy

import DeepCardioFunctions as dc
import monkey_functions as test

import matplotlib as mpl
import pandas as pd

local_path    = os.getcwd()
cases_folder  = local_path + '/Synthetic_shapes/'
case_name     = "Test"
POD_folder_4D = local_path  + '/Functional_model/'
out_folder    = cases_folder + case_name + '/PINN_data/'


# Create a new session
with tf.Session() as sess:
    # Load the saved model
    saver = tf.train.import_meta_graph(out_folder + '/Trained_model.ckpt.meta')
    
    # Restore the variables
    saver.restore(sess, out_folder + '/Trained_model.ckpt')
    
    # Get the graph
    graph = tf.get_default_graph()

    p_tf = graph.get_tensor_by_name("input_placeholder:0")
    print(p_tf)
    a_pred = graph.get_tensor_by_name("output_prediction:0")
    print(a_pred)