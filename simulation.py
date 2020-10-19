import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy import stats
import pandas as pd
from operator import add 
import itertools
import sklearn
from sklearn import preprocessing
from scipy import stats
import copy

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader


import code_paper_submit
from code_paper_submit import sim_graph_binaryZ0
from code_paper_submit import PrepareData_test
from code_paper_submit import run_iters_rep


# simulation parameters
nz = 3
ny1 = 9
n_cat = [2,2,2, 3,3,3, 4,4,4]
a = [ np.array([-1]), np.array([-0.5]), np.array([-1]), 
      np.array([-1,-2]), np.array([-0.5,-1]), np.array([-1,-1]), 
      np.array([0.5,-0.5,-1]), np.array([-0.5,-1,0.5]), np.array([0.5,-1,-2]) ]
W = [ np.array([[3], [1], [2]]), 
      np.array([[1],[-1], [0]]),   
      np.array([[0],[0], [0]]),    
      np.array([[1,1], [-1,-2], [0,0]]),
      np.array([[0,0], [0,0], [0,0]]),
      np.array([[0, 1], [2,4], [1,2]]),       
      np.array([[0,1,1.5], [0,0,1], [1,2,3]]),      
      np.array([[0,0,0], [2,1,0], [1,-1,-2]]),
      np.array([[2,0,-2], [0,0,0], [0,0,-2]]) ]
ny2 = 5
a2 = np.array([1, 2, 0, -1, -2])
W2 = np.array([[1,1,0,-1,-2], [2,2,-2,0,-1], [1,2,0,-1,1]])
nx = 2
theta0 = np.array([2, -1, 1])  
theta1 = np.array([[1,0.5,0], [0,1,0], [0,0.5,2]])
theta2 = np.array([[1,0.5,0],  [0.5,-1,0]])
theta3 = np.array([-0.5, -0.5, 1])
theta4 = np.array([[1,0,-2],  [-0.5,3,-1],  [0,-2,2]])
theta5 = np.array([[2,-0.5,1],  [-1,2,1]])




#### simualte test data
n_test = 100000 
data_t = sim_graph_binaryZ0(111, n_test, ny1, nz, nx, n_cat, a, W, theta0, theta1, theta2, theta3, theta4, theta5, ny2, a2, W2)
dataset_t = PrepareData_test (data_t['X'], ny1, data_t['y0_obs'], data_t['z0'], data_t['y0_obs_c'])  
data_t['y1sum_sel'] = np.sum(data_t['y1_obs'][:,[0,5,6]], axis=1) + np.sum(data_t['y1_obs_c'][:,[0,1]], axis=1)


# run the simulation, and print out the value functions & accuracy.
nobs = 200
sim_res = run_iters_rep(0, 1, nobs, nz, ny1, n_cat=n_cat, nx=nx, 
                a=a, W=W, theta0=theta0, theta1=theta1, theta2=theta2, theta3=theta3, theta4=theta4, theta5=theta5, 
                ny2=ny2, a2=a2, W2=W2,
                nV=10, V_act="sigmoid", nV2=20, V2_act="sigmoid", V_drop=None, V2_drop=None, noV=False,
                niter_model=6, niter_whole=6,  batch_size=100, optimizer="adam", lrate=0.1, lr_decay=1e-6, 
                momentum=0.5, nesterov=True, weight_decay=0, loss_weights=None, validation_split = 0, 
                cal_loss=True, show_fig = False, print_model=False, pred_trt=True,  
                data=None,  dataset_t=dataset_t, n_test=n_test, data_t=data_t, 
                init_weight=True, set_zero_grad=False,  z_weight=None, 
                run_owls=False, kernel=None, c=None, gamma=None, 
                run_ours=True)