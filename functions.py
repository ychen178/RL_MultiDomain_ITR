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



### Define the model class -- "Net"
class Net(nn.Module):
    
    def __init__(self, nz, n_cat, nx, nV, nV2=None, noV=False, ny2=None):
        super().__init__()
        
        self.ny1 = np.shape(n_cat)[0]  

        # Z to categorical Y's
        temp_z_to_y = list()
        for j in range(self.ny1):
            temp_z_to_y.append(nn.Linear(nz, n_cat[j]))
        self.z_to_y = nn.ModuleList(temp_z_to_y)
        
        # Z to continuous Y's
        if ny2 is not None:
            self.z_to_y_c = nn.Linear(nz, ny2)
        
        if noV is False:  # there is V layer
            if nV2 is None: 
                self.V = nn.Linear(nz+nx, nV)  # V always connect to z1
            else:
                self.V2 = nn.Linear(nz+nx, nV2)
                self.V = nn.Linear(nV2, nV)
            self.z1_m1 = nn.Linear(nV, nz)
            self.z1_m0 = nn.Linear(nV, nz)    
        else:  # there is no V layer
            self.z1_m1 = nn.Linear(nz+nx, nz) # directly to Z1
            self.z1_m0 = nn.Linear(nz+nx, nz)   

    def forward(self, x, z0, A, V_act, V2_act=None, V_drop=None, V2_drop=None, noV=False): 
        
        y0 = list()
        for j in range(self.ny1):
            y0.append(F.log_softmax(self.z_to_y[j](z0), dim=1))
            
        if hasattr(self, 'z_to_y_c'):
            y0_c = self.z_to_y_c(z0)
            
        x_z0 = torch.cat((x, z0), 1)
            
        if noV is False:  # there is layer V
            if V2_act is None: 
                if V_act is "relu":
                    V = F.relu(self.V(x_z0))
                elif V_act is "sigmoid":
                    V = torch.sigmoid(self.V(x_z0))
                if V_drop is not None:
                    V = F.dropout(V, p=V_drop)
            else:  # make same activation as V_act
                if V_act is "relu":
                    V2 = F.relu(self.V2(x_z0))
                    if V2_drop is not None:
                        V2 = F.dropout(V2, p=V2_drop)
                    V = F.relu(self.V(V2))
                elif V_act is "sigmoid":
                    V2 = torch.sigmoid(self.V2(x_z0))
                    if V2_drop is not None:
                        V2 = F.dropout(V2, p=V2_drop)
                    V = torch.sigmoid(self.V(V2))
                if V_drop is not None:
                    V = F.dropout(V, p=V_drop)

            # from V to Z1, for A=1 and A=-1 separately
            temp_z1_m1 = torch.sigmoid(self.z1_m1(V))
            temp_z1_m1_c = temp_z1_m1.clone()
            temp_z1_m1_c[A==-1,] = 0  

            temp_z1_m0 = torch.sigmoid(self.z1_m0(V))
            temp_z1_m0_c = temp_z1_m0.clone()
            temp_z1_m0_c[A==1,] = 0
        
        else:   # no V layer, X z0 directly to Z1
            temp_z1_m1 = torch.sigmoid(self.z1_m1(x_z0)) # output probability
            temp_z1_m1_c = temp_z1_m1.clone()
            temp_z1_m1_c[A==-1,] = 0  

            temp_z1_m0 = torch.sigmoid(self.z1_m0(x_z0))
            temp_z1_m0_c = temp_z1_m0.clone()
            temp_z1_m0_c[A==1,] = 0
            
        z1 = temp_z1_m1_c + temp_z1_m0_c
        
        y1 = list()
        for j in range(self.ny1): 
            y1.append(F.log_softmax(self.z_to_y[j](z1), dim=1))
        
        if hasattr(self, 'z_to_y_c'):
            y1_c = self.z_to_y_c(z1)
            return y0, y1, z1, y0_c, y1_c
        else:
            return y0, y1, z1, None, None

        
        
###  PrepareData function --- for training data
class PrepareData(Dataset): 

    def __init__(self, X, A, z0, ny1, y01, y11, y02=None, y12=None):
        
        self.ny1 = ny1
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X).float()
        if not torch.is_tensor(A):
            self.A = torch.from_numpy(A).float()
        if not torch.is_tensor(z0):
            self.z0 = torch.from_numpy(z0).float()
        if not torch.is_tensor(y01):
            self.y0 = []   
            self.y1 = []  
            for j in range(ny1):
                self.y0.append(torch.from_numpy(y01[:,j]).long())
                self.y1.append(torch.from_numpy(y11[:,j]).long())   
        if y02 is not None:
            self.y0_c = torch.from_numpy(y02).float()
        if y12 is not None:
            self.y1_c = torch.from_numpy(y12).float()    

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y0_idx = []
        y1_idx = []
        for j in range(self.ny1):
            y0_idx.append(self.y0[j][idx]) 
            y1_idx.append(self.y1[j][idx])
        if hasattr(self, 'y0_c') and hasattr(self, 'y1_c'):
            return (self.X[idx], self.z0[idx], self.A[idx], y0_idx, y1_idx, self.y0_c[idx], self.y1_c[idx])  
        else:
            return (self.X[idx], self.z0[idx], self.A[idx], y0_idx, y1_idx, 'none', 'none')

        
        
        
###  PrepareData_test function --- used to prepare testset data (only have X, Y0 input)
class PrepareData_test(Dataset):

    def __init__(self, X, ny1, y01, z0=None, y02=None):
        
        self.ny1 = ny1
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X).float()
        if z0 is not None:
            self.z0 = torch.from_numpy(z0).float()
        else: 
            self.z0 = None
        if not torch.is_tensor(y01):
            self.y0 = []
            for j in range(ny1):
                self.y0.append(torch.from_numpy(y01[:,j]).long())  
        if y02 is not None: 
            self.y0_c = torch.from_numpy(y02).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):  
        y0_idx = []
        for j in range(self.ny1):
            y0_idx.append(self.y0[j][idx]) 
        if self.z0 is not None:
            z0_idx = self.z0[idx]
        else:
            z0_idx = 'none'
        if hasattr(self, 'y0_c'):
            return (self.X[idx], y0_idx, z0_idx, self.y0_c[idx])
        else:
            return (self.X[idx], y0_idx, z0_idx, 'none')
        
        
        
###  run_net: run the model
def run_net(nz, ny1, n_cat, y01, y11, X, A, z0=None, ny2=None, y02=None, y12=None, 
            nV=10, V_act="relu",  nV2=None, V2_act=None, V_drop=0.5, V2_drop=0.5, noV=False,
            niter=2, batch_size=50, optimizer="adam", lrate=0.1, lr_decay=1e-6, momentum=0.5, weight_decay=0, loss_weights=None, validation_split = 0.2, 
            cal_loss=True, show_fig=True, print_model=True, init_weight=False, set_zero_grad=False):
    
    if y01.shape != y11.shape:
        raise ValueError("dimension of y01 and y11 differ")

    nobs = y01.shape[0]
    nx = X.shape[1]
    
    if z0 is None:
        z0 = np.random.binomial(n=1, p=0.5, size=(nobs, nz))     
 
    net = Net(nz, n_cat, nx, nV, nV2, noV, ny2) 
    
    if init_weight is not False:
        net = init_weights(net) 
           
    if print_model is True:
        print(net)
    
    loss_function = nn.NLLLoss()
    loss_function_c = nn.MSELoss()
    
    if optimizer is "adam":
        opt = optim.Adam(net.parameters(), lr=lrate, weight_decay=weight_decay)
    elif optimizer is "sgd":
        opt = optim.SGD(net.parameters(), lr=lrate, momentum=momentum, weight_decay=weight_decay)
        
    if validation_split is not None and validation_split > 0:
        n_train = round(nobs * (1-validation_split))
        n_test = nobs - n_train
    else:
        n_train = nobs
        
    if y02 is not None and y12 is not None:
        ds_train = PrepareData(X[0: n_train,], A[0: n_train], z0[0: n_train,], ny1, y01[0: n_train,], y11[0: n_train,], y02[0: n_train,], y12[0: n_train,])
        if validation_split is not None and validation_split > 0: 
            ds_test = PrepareData(X[n_train: nobs,], A[n_train: nobs], z0[n_train: nobs,], ny1, y01[n_train: nobs,], y11[n_train: nobs,], y02[n_train: nobs,], y12[n_train: nobs,])   
    else:
        ds_train = PrepareData(X[0: n_train,], A[0: n_train], z0[0: n_train,], ny1, y01[0: n_train,], y11[0: n_train,])
        if validation_split is not None and validation_split > 0: 
            ds_test = PrepareData(X[n_train: nobs,], A[n_train: nobs], z0[n_train: nobs,], ny1, y01[n_train: nobs,], y11[n_train: nobs,])
    
    ds_tr = DataLoader(ds_train, batch_size=n_train, shuffle=False)
    ds_tr_batch = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    if validation_split is not None and validation_split > 0: 
        ds_t = DataLoader(ds_test, batch_size=n_test, shuffle=False)

    ds_all = PrepareData(X, A, z0, ny1, y01, y11, y02, y12)
    ds_all_loader = DataLoader(ds_all, batch_size=nobs, shuffle=False)
     
    # record loss and accuracy from all iterations -- training & validation
    losses_all = []
    accuracy_all = []
    losses_all_t = []
    accuracy_all_t = []
    mse_all = []
    mse_all_t = []
    
    for epoch in range(niter):
        for idx, (X_torch_batch, z0_torch_batch, A_torch_batch, y0_torch_batch, y1_torch_batch, y0_c_torch_batch, y1_c_torch_batch) in enumerate(ds_tr_batch):
        
            net.zero_grad()
            output_y0_batch, output_y1_batch, output_z1_batch, output_y0_c_batch, output_y1_c_batch = net(X_torch_batch, z0_torch_batch, A_torch_batch, V_act, V2_act, V_drop, V2_drop, noV)
            
            loss_y0_batch = []
            loss_y1_batch = []
            for j in range(ny1):
                loss_y0_batch.append(loss_function(output_y0_batch[j], y0_torch_batch[j]))  # use "[j]" since y0_torch is a list 
                loss_y1_batch.append(loss_function(output_y1_batch[j], y1_torch_batch[j]))
            
            if ny2 is not None: 
                mse_y0_batch = []
                mse_y1_batch = []  
                for j in range(ny2):
                    mse_y0_batch.append(loss_function_c(output_y0_c_batch[:,j], y0_c_torch_batch[:,j]))  # take the jth column
                    mse_y1_batch.append(loss_function_c(output_y0_c_batch[:,j], y0_c_torch_batch[:,j]))
                if loss_weights is not None:
                    loss_batch = sum(loss_y0_batch)*loss_weights[0] + sum(loss_y1_batch)*loss_weights[1] + sum(mse_y0_batch)*loss_weights[2] + sum(mse_y1_batch)*loss_weights[3]
                else:
                    loss_batch = sum(loss_y0_batch) + sum(loss_y1_batch) + sum(mse_y0_batch) + sum(mse_y1_batch)
            else:
                if loss_weights is not None:
                    loss_batch = sum(loss_y0_batch)*loss_weights[0] + sum(loss_y1_batch)*loss_weights[1]
                else:
                    loss_batch = sum(loss_y0_batch) + sum(loss_y1_batch)
            
            #-------  calcualte loss for the training set
            if cal_loss is True:
                for idx, (X_torch, z0_torch, A_torch, y0_torch, y1_torch, y0_c_torch, y1_c_torch) in enumerate(ds_tr): # this is the whole dataset
                    output_y0, output_y1, output_z1, output_y0_c, output_y1_c = net(X_torch, z0_torch, A_torch, V_act, V2_act, V_drop, V2_drop, noV)
            
                    loss_y0 = []
                    loss_y1 = []
                    pred_y0 = []
                    pred_y1 = []
                    accuracy_y0 = []
                    accuracy_y1 = []
                    for j in range(ny1):
                        loss_y0.append(loss_function(output_y0[j], y0_torch[j]))   
                        loss_y1.append(loss_function(output_y1[j], y1_torch[j]))

                        y0_temp = torch.argmax(output_y0[j], dim=1) 
                        y1_temp = torch.argmax(output_y1[j], dim=1)
                        pred_y0.append(y0_temp)
                        pred_y1.append(y1_temp)
                        accuracy_y0.append(np.mean((y0_temp==y0_torch[j]).detach().numpy()))
                        accuracy_y1.append(np.mean((y1_temp==y1_torch[j]).detach().numpy()))
                     
                    mse_y0_c = []
                    mse_y1_c = []
                    if ny2 is not None:
                        for j in range(ny2):
                            mse_y0_c.append(loss_function_c(output_y0_c[:,j], y0_c_torch[:,j]))
                            mse_y1_c.append(loss_function_c(output_y1_c[:,j], y1_c_torch[:,j]))
                        mse = sum(mse_y0_c) + sum(mse_y1_c)
                        loss = sum(loss_y0) + sum(loss_y1) + mse
                        mse_all.append(np.asscalar(mse.detach().numpy()))
                    else:
                        loss = sum(loss_y0) + sum(loss_y1)
                        
                    losses_all.append(np.asscalar(loss.detach().numpy()))
                    accuracy = np.mean([accuracy_y0, accuracy_y1])
                    accuracy_all.append(accuracy)
                    
            #-------  calculate loss for the validation set
            if validation_split is not None and validation_split > 0 and cal_loss is True:
                for idx, (X_torch_t, z0_torch_t, A_torch_t, y0_torch_t, y1_torch_t, y0_c_torch_t, y1_c_torch_t) in enumerate(ds_t): # this is the whole dataset
                    output_y0_t, output_y1_t, output_z1_t, output_y0_c_t, output_y1_c_t = net(X_torch_t, z0_torch_t, A_torch_t, V_act, V2_act, V_drop, V2_drop, noV) 

                    loss_y0_t = [] 
                    loss_y1_t = []
                    pred_y0_t = []
                    pred_y1_t = []
                    accuracy_y0_t = []
                    accuracy_y1_t = []
                    for j in range(ny1):
                        loss_y0_t.append(loss_function(output_y0_t[j], y0_torch_t[j]))
                        loss_y1_t.append(loss_function(output_y1_t[j], y1_torch_t[j]))
                        y0_temp_t = torch.argmax(output_y0_t[j], dim=1) 
                        y1_temp_t = torch.argmax(output_y1_t[j], dim=1)
                        pred_y0_t.append(y0_temp_t)
                        pred_y1_t.append(y1_temp_t)
                        accuracy_y0_t.append(np.mean((y0_temp_t==y0_torch_t[j]).detach().numpy()))
                        accuracy_y1_t.append(np.mean((y1_temp_t==y1_torch_t[j]).detach().numpy()))

                    mse_y0_c_t = []
                    mse_y1_c_t = []
                    if ny2 is not None:
                        for j in range(ny2):
                            mse_y0_c_t.append(loss_function_c(output_y0_c_t[:,j], y0_c_torch_t[:,j]))
                            mse_y1_c_t.append(loss_function_c(output_y1_c_t[:,j], y1_c_torch_t[:,j]))
                        mse_t = sum(mse_y0_c_t) + sum(mse_y1_c_t)
                        loss_t = sum(loss_y0_t) + sum(loss_y1_t) + mse_t
                        mse_all_t.append(mse_t.detach().numpy())  # append all 
                    else: 
                        loss_t = sum(loss_y0_t) + sum(loss_y1_t)
                        
                    losses_all_t.append(np.asscalar(loss_t.detach().numpy()))  # record loss
                    accuracy_t = np.mean([np.mean(accuracy_y0_t), np.mean(accuracy_y1_t)])
                    accuracy_all_t.append(accuracy_t)  # record accuracy

            loss_batch.backward()  
            # do not update certain gradients
            if set_zero_grad is not False:
                net = set_zero_grads(net)
            
            opt.step()  
    
        #-------- output the last iteration individual loss and accuracy for each Y (training & test set)
        loss_y0_last = np.zeros(shape=(ny1))
        loss_y1_last = np.zeros(shape=(ny1))
        accuracy_y0_last = np.zeros(shape=(ny1))
        accuracy_y1_last = np.zeros(shape=(ny1))
        # for validation set
        if validation_split is not None and validation_split > 0:
            loss_y0_last_t = np.zeros(shape=(ny1))
            loss_y1_last_t = np.zeros(shape=(ny1))
            accuracy_y0_last_t = np.zeros(shape=(ny1))
            accuracy_y1_last_t = np.zeros(shape=(ny1))
        else:
            loss_y0_last_t = None
            loss_y1_last_t = None
            accuracy_y0_last_t = None
            accuracy_y1_last_t = None 
        
        for j in range(ny1):
            loss_y0_last[j] = loss_y0[j].detach().numpy()
            loss_y1_last[j] = loss_y1[j].detach().numpy()
            accuracy_y0_last[j] = accuracy_y0[j]
            accuracy_y1_last[j] = accuracy_y1[j]
            if validation_split is not None and validation_split > 0:
                loss_y0_last_t[j] = loss_y0_t[j].detach().numpy()
                loss_y1_last_t[j] = loss_y1_t[j].detach().numpy()
                accuracy_y0_last_t[j] = accuracy_y0_t[j]
                accuracy_y1_last_t[j] = accuracy_y1_t[j]
            
        if ny2 is not None:
            mse_y0_c_last = np.zeros(shape=(ny2))
            mse_y1_c_last = np.zeros(shape=(ny2))
            if validation_split is not None and validation_split > 0:
                mse_y0_c_t_last = np.zeros(shape=(ny2))
                mse_y1_c_t_last = np.zeros(shape=(ny2))
            else:
                mse_y0_c_t_last = None
                mse_y1_c_t_last = None
            for j in range(ny2):
                mse_y0_c_last[j] = mse_y0_c[j].detach().numpy()
                mse_y1_c_last[j] = mse_y1_c[j].detach().numpy() 
                if validation_split is not None and validation_split > 0:
                    mse_y0_c_t_last[j] = mse_y0_c_t[j].detach().numpy() 
                    mse_y1_c_t_last[j] = mse_y0_c_t[j].detach().numpy() 
        else:
            mse_y0_c_last = None
            mse_y1_c_last = None
            mse_y0_c_t_last = None
            mse_y1_c_t_last = None
        
        if validation_split is not None and validation_split > 0:
            output_z1_t = output_z1_t.detach().numpy() 
        else: 
            output_z1_t = None
            ds_test = None
            
    if show_fig is True: 
        plt.plot(losses_all)
        plt.plot(losses_all_t)
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()    

        plt.plot(accuracy_all)
        plt.plot(accuracy_all_t)
        plt.title('Accuracy for categorical Ys')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()    
        if ny2 is not None:
            plt.plot(mse_all)  
            plt.plot(mse_all_t)
            plt.title('MSE for continuous Ys')
            plt.ylabel('MSE for continuous Ys')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()   
    

    return{'model':net, 'loss':losses_all, 'accuracy':accuracy_all, 'mse':mse_all, 
           'loss_t':losses_all_t, 'accuracy_t':accuracy_all_t, 'mse_t':mse_all_t,  # the above are to record for from all minibatches
           'loss_y0':loss_y0_last, 'loss_y1':loss_y1_last, 'accuracy_y0':accuracy_y0_last, 'accuracy_y1':accuracy_y1_last,
           'loss_y0_t':loss_y0_last_t, 'loss_y1_t':loss_y1_last_t, 'accuracy_y0_t':accuracy_y0_last_t, 'accuracy_y1_t':accuracy_y1_last_t, 
           'mse_y0_c':mse_y0_c_last, 'mse_y1_c':mse_y0_c_last, 'mse_y0_c_t':mse_y0_c_t_last, 'mse_y1_c_t':mse_y1_c_t_last,
           'nz':nz, 'ny1':ny1, 'n_cat':n_cat, 'ny2':ny2, 'optimizer': optimizer, 'nV': nV, 'V_act': V_act, 
           'nV2': nV2, 'V2_act': V2_act, 'V_drop':V_drop, 'V2_drop':V2_drop, 'noV':noV,  'set_zero_grad':set_zero_grad,
           'loss_weights': loss_weights, # add on 0114
           'z1': output_z1.detach().numpy(), 'z1_t': output_z1_t,
           'validation_split':validation_split, 
           'weight_decay':weight_decay, 'batch_size': batch_size, 'lrate':lrate, 'lr_decay': lr_decay, 'momentum':momentum,
           'nobs': nobs, 'ds_train': ds_train, 'ds_test': ds_test, 'ds_all': ds_all} # all are Datasets, not DataLoader




###  Define a function to continue running from the current model with the new z0
def run_next_net(z0, run_net_object, niter=2, batch_size=None, lrate=None, lr_decay=None, momentum=None,
                 cal_loss=True, show_fig=True, print_model=False):
    
    net = run_net_object['model']
    if print_model is True:
        print(net)
    
    nz = run_net_object['nz']
    ny1 = run_net_object['ny1']
    ny2 = run_net_object['ny2']
    n_cat = run_net_object['n_cat']
    optimizer = run_net_object['optimizer']
    weight_decay = run_net_object['weight_decay']
    validation_split = run_net_object['validation_split']
    nobs = run_net_object['nobs']
    nV = run_net_object['nV']
    V_act = run_net_object['V_act']
    nV2 = run_net_object['nV2']
    V2_act = run_net_object['V2_act']
    V_drop = run_net_object['V_drop']
    V2_drop = run_net_object['V2_drop']
    noV = run_net_object['noV']
    set_zero_grad = run_net_object['set_zero_grad'] # added on 0106
    loss_weights = run_net_object['loss_weights']   # added on 0114
    
    if batch_size is None:
        batch_size = run_net_object['batch_size']
    if lrate is None:
        lrate = run_net_object['lrate']
    if lr_decay is None:
        lr_decay = run_net_object['lr_decay']
    if momentum is None:
        momentum = run_net_object['momentum']

    loss_function = nn.NLLLoss() 
    loss_function_c = nn.MSELoss()
    
    if optimizer is "adam":
        opt = optim.Adam(net.parameters(), lr=lrate, weight_decay=weight_decay)
    elif optimizer is "sgd":
        opt = optim.SGD(net.parameters(), lr=lrate, momentum=momentum, weight_decay=weight_decay)
        
    n_train = round(nobs * (1 - validation_split))
    n_test = nobs - n_train

    z0_train_torch = torch.from_numpy(z0[0:n_train, ]).float()
    z0_test_torch = torch.from_numpy(z0[n_train:nobs, ]).float()
    z0_all_torch = torch.from_numpy(z0).float()
        
    ds_train = run_net_object['ds_train']
    ds_test = run_net_object['ds_test']
    ds_all = run_net_object['ds_all']
    
    ds_train.z0 = z0_train_torch
    if ds_test is not None:
        ds_test.z0 = z0_test_torch
    ds_all.z0 = z0_all_torch
    
    ds_tr_batch = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    ds_tr = DataLoader(ds_train, batch_size=n_train, shuffle=False)
    if ds_test is not None:
        ds_t = DataLoader(ds_test, batch_size=n_test, shuffle=False)
    
    losses_all = run_net_object['loss']
    accuracy_all = run_net_object['accuracy']
    losses_all_t = run_net_object['loss_t']
    accuracy_all_t = run_net_object['accuracy_t']
    mse_all = run_net_object['mse']
    mse_all_t = run_net_object['mse_t']
    
    for epoch in range(niter):
        for idx, (X_torch_batch, z0_torch_batch, A_torch_batch, y0_torch_batch, y1_torch_batch, y0_c_torch_batch, y1_c_torch_batch) in enumerate(ds_tr_batch):
            net.zero_grad()
            output_y0_batch, output_y1_batch, output_z1_batch, output_y0_c_batch, output_y1_c_batch = net(X_torch_batch, z0_torch_batch, A_torch_batch, V_act, V2_act, V_drop, V2_drop, noV)
            
            loss_y0_batch = []
            loss_y1_batch = []
            for j in range(ny1):
                loss_y0_batch.append(loss_function(output_y0_batch[j], y0_torch_batch[j]))  # use "[j]" since y0_torch is a list 
                loss_y1_batch.append(loss_function(output_y1_batch[j], y1_torch_batch[j]))
            
            if ny2 is not None:
                mse_y0_batch = []
                mse_y1_batch = []  
                for j in range(ny2):
                    mse_y0_batch.append(loss_function_c(output_y0_c_batch[:,j], y0_c_torch_batch[:,j]))  # take the jth column
                    mse_y1_batch.append(loss_function_c(output_y0_c_batch[:,j], y0_c_torch_batch[:,j]))
                loss_batch = sum(loss_y0_batch) + sum(loss_y1_batch) + sum(mse_y0_batch) + sum(mse_y1_batch)
            else:
                loss_batch = sum(loss_y0_batch) + sum(loss_y1_batch)
            
            #-------  calcualte loss for the training set
            if cal_loss is True:
                for idx, (X_torch, z0_torch, A_torch, y0_torch, y1_torch, y0_c_torch, y1_c_torch) in enumerate(ds_tr): # this is the whole dataset
                    output_y0, output_y1, output_z1, output_y0_c, output_y1_c = net(X_torch, z0_torch, A_torch, V_act, V2_act, V_drop, V2_drop, noV)  # added V2_act
            
                    loss_y0 = []
                    loss_y1 = []
                    pred_y0 = []
                    pred_y1 = []
                    accuracy_y0 = []
                    accuracy_y1 = []
                    for j in range(ny1):
                        loss_y0.append(loss_function(output_y0[j], y0_torch[j]))   
                        loss_y1.append(loss_function(output_y1[j], y1_torch[j]))

                        y0_temp = torch.argmax(output_y0[j], dim=1) 
                        y1_temp = torch.argmax(output_y1[j], dim=1)
                        pred_y0.append(y0_temp)
                        pred_y1.append(y1_temp)
                        accuracy_y0.append(np.mean((y0_temp==y0_torch[j]).detach().numpy()))
                        accuracy_y1.append(np.mean((y1_temp==y1_torch[j]).detach().numpy()))
                     
                    mse_y0_c = []
                    mse_y1_c = []
                    if ny2 is not None:
                        for j in range(ny2):
                            mse_y0_c.append(loss_function_c(output_y0_c[:,j], y0_c_torch[:,j]))
                            mse_y1_c.append(loss_function_c(output_y1_c[:,j], y1_c_torch[:,j]))
                        mse = sum(mse_y0_c) + sum(mse_y1_c)
                        loss = sum(loss_y0) + sum(loss_y1) + mse
                        mse_all.append(np.asscalar(mse.detach().numpy()))
                    else:
                        loss = sum(loss_y0) + sum(loss_y1)
                        
                    losses_all.append(np.asscalar(loss.detach().numpy()))
                    accuracy = np.mean([accuracy_y0, accuracy_y1])
                    accuracy_all.append(accuracy)
                    
            #-------  calculate loss for the validation set
            if validation_split is not None and validation_split > 0 and cal_loss is True:
                for idx, (X_torch_t, z0_torch_t, A_torch_t, y0_torch_t, y1_torch_t, y0_c_torch_t, y1_c_torch_t) in enumerate(ds_t):
                    output_y0_t, output_y1_t, output_z1_t, output_y0_c_t, output_y1_c_t = net(X_torch_t, z0_torch_t, A_torch_t, V_act, V2_act, V_drop, V2_drop, noV)

                    loss_y0_t = []
                    loss_y1_t = []
                    pred_y0_t = []
                    pred_y1_t = []
                    accuracy_y0_t = []
                    accuracy_y1_t = []
                    for j in range(ny1):
                        loss_y0_t.append(loss_function(output_y0_t[j], y0_torch_t[j]))
                        loss_y1_t.append(loss_function(output_y1_t[j], y1_torch_t[j]))
                        y0_temp_t = torch.argmax(output_y0_t[j], dim=1) 
                        y1_temp_t = torch.argmax(output_y1_t[j], dim=1)
                        pred_y0_t.append(y0_temp_t)
                        pred_y1_t.append(y1_temp_t)
                        accuracy_y0_t.append(np.mean((y0_temp_t==y0_torch_t[j]).detach().numpy()))
                        accuracy_y1_t.append(np.mean((y1_temp_t==y1_torch_t[j]).detach().numpy()))

                    mse_y0_c_t = []
                    mse_y1_c_t = []
                    if ny2 is not None:
                        for j in range(ny2):
                            mse_y0_c_t.append(loss_function_c(output_y0_c_t[:,j], y0_c_torch_t[:,j]))
                            mse_y1_c_t.append(loss_function_c(output_y1_c_t[:,j], y1_c_torch_t[:,j]))
                        mse_t = sum(mse_y0_c_t) + sum(mse_y1_c_t)
                        loss_t = sum(loss_y0_t) + sum(loss_y1_t) + mse_t
                        mse_all_t.append(mse_t.detach().numpy())
                    else: 
                        loss_t = sum(loss_y0_t) + sum(loss_y1_t)
                        
                    losses_all_t.append(np.asscalar(loss_t.detach().numpy()))  # record loss
                    accuracy_t = np.mean([np.mean(accuracy_y0_t), np.mean(accuracy_y1_t)])
                    accuracy_all_t.append(accuracy_t)  # record accuracy

            loss_batch.backward()  
            # do not update certain gradients
            if set_zero_grad is not False:
                net = set_zero_grads(net) 
                
            opt.step()  

        #-------- output the last iteration individual loss and accuracy for each Y (training & test set)
        loss_y0_last = np.zeros(shape=(ny1))
        loss_y1_last = np.zeros(shape=(ny1))
        accuracy_y0_last = np.zeros(shape=(ny1))
        accuracy_y1_last = np.zeros(shape=(ny1))
        # for validation set
        if validation_split is not None and validation_split > 0:
            loss_y0_last_t = np.zeros(shape=(ny1))
            loss_y1_last_t = np.zeros(shape=(ny1))
            accuracy_y0_last_t = np.zeros(shape=(ny1))
            accuracy_y1_last_t = np.zeros(shape=(ny1))
        else:
            loss_y0_last_t = None
            loss_y1_last_t = None
            accuracy_y0_last_t = None
            accuracy_y1_last_t = None 
        
        for j in range(ny1):
            loss_y0_last[j] = loss_y0[j].detach().numpy()
            loss_y1_last[j] = loss_y1[j].detach().numpy()
            accuracy_y0_last[j] = accuracy_y0[j]
            accuracy_y1_last[j] = accuracy_y1[j]
            if validation_split is not None and validation_split > 0:
                loss_y0_last_t[j] = loss_y0_t[j].detach().numpy()
                loss_y1_last_t[j] = loss_y1_t[j].detach().numpy()
                accuracy_y0_last_t[j] = accuracy_y0_t[j]
                accuracy_y1_last_t[j] = accuracy_y1_t[j]
            
        if ny2 is not None:
            mse_y0_c_last = np.zeros(shape=(ny2))
            mse_y1_c_last = np.zeros(shape=(ny2))
            if validation_split is not None and validation_split > 0:
                mse_y0_c_t_last = np.zeros(shape=(ny2))
                mse_y1_c_t_last = np.zeros(shape=(ny2))
            else:
                mse_y0_c_t_last = None
                mse_y1_c_t_last = None
            for j in range(ny2):
                mse_y0_c_last[j] = mse_y0_c[j].detach().numpy()
                mse_y1_c_last[j] = mse_y1_c[j].detach().numpy() 
                if validation_split is not None and validation_split > 0:
                    mse_y0_c_t_last[j] = mse_y0_c_t[j].detach().numpy() 
                    mse_y1_c_t_last[j] = mse_y0_c_t[j].detach().numpy() 
        else:
            mse_y0_c_last = None
            mse_y1_c_last = None
            mse_y0_c_t_last = None
            mse_y1_c_t_last = None
        
        if validation_split is not None and validation_split > 0:  # for the ease of output
            output_z1_t = output_z1_t.detach().numpy() 
        else: 
            output_z1_t = None
            ds_test = None            
            
    if show_fig is True: 
        # plot loss
        plt.plot(losses_all)
        plt.plot(losses_all_t)
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()    
        # plot accuracy
        plt.plot(accuracy_all)
        plt.plot(accuracy_all_t)
        plt.title('Accuracy for categorical Ys')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()    
        if ny2 is not None:
            plt.plot(mse_all)
            plt.plot(mse_all_t)
            plt.title('MSE for continuous Ys')
            plt.ylabel('MSE for continuous Ys')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()   
    
    return{'model':net, 'loss':losses_all, 'accuracy':accuracy_all, 'mse':mse_all, 
           'loss_t':losses_all_t, 'accuracy_t':accuracy_all_t, 'mse_t':mse_all_t,  # the above are to record for from all minibatches
           'loss_y0':loss_y0_last, 'loss_y1':loss_y1_last, 'accuracy_y0':accuracy_y0_last, 'accuracy_y1':accuracy_y1_last,
           'loss_y0_t':loss_y0_last_t, 'loss_y1_t':loss_y1_last_t, 'accuracy_y0_t':accuracy_y0_last_t, 'accuracy_y1_t':accuracy_y1_last_t, 
           'mse_y0_c':mse_y0_c_last, 'mse_y1_c':mse_y0_c_last, 'mse_y0_c_t':mse_y0_c_t_last, 'mse_y1_c_t':mse_y1_c_t_last,
           'nz':nz, 'ny1':ny1, 'n_cat':n_cat, 'ny2':ny2, 'optimizer': optimizer, 'nV': nV, 'V_act': V_act, 
           'nV2': nV2, 'V2_act': V2_act, 'V_drop':V_drop, 'V2_drop':V2_drop, 'noV':noV,  'set_zero_grad':set_zero_grad,  # don't foget set_zero_grad !!! added on 0106
           'loss_weights': loss_weights, # add on 0114
           'z1': output_z1.detach().numpy(), 'z1_t': output_z1_t, # added
           'validation_split':validation_split, 
           'weight_decay':weight_decay, 'batch_size': batch_size, 'lrate':lrate, 'lr_decay': lr_decay, 'momentum':momentum,
           'nobs': nobs, 'ds_train': ds_train, 'ds_test': ds_test, 'ds_all': ds_all} # all are Datasets, not DataLoader




###  find best_z0 for all subjects at the same time in model training --  minimal loss in all Y0's & Y1's
def best_z0_all (run_net_object, nobs, dataset):  # newly defined function inputs
    
    model = run_net_object['model']
    nz = run_net_object['nz']
    ny1 = run_net_object['ny1']
    ny2 = run_net_object['ny2'] 
    n_cat = run_net_object['n_cat']

    V_act = run_net_object['V_act']
    V2_act = run_net_object['V2_act']
    V_drop = run_net_object['V_drop']
    V2_drop = run_net_object['V2_drop']
    noV = run_net_object['noV']
    loss_weights = run_net_object['loss_weights'] 
    
    dat_loader = DataLoader(dataset, batch_size=nobs, shuffle=False)
    for idx, (X_torch, z0_torch, A_torch, y0_torch, y1_torch, y0_c_torch, y1_c_torch) in enumerate(dat_loader): # this is the whole dataset. z0_torch not used

        min_loss = np.array([math.inf] * nobs)   
        z_best = np.zeros(shape=(nobs, nz))      

        loss_function = nn.NLLLoss(reduction='none')  
        loss_function_c = nn.MSELoss(reduction='none')
    
        # loop over all Zs to calculate individual loss over all Y0's and Y1's
        for z in itertools.product([0,1], repeat=nz):   
            z_rep = np.array([z]*nobs)
            z_torch = torch.from_numpy(z_rep).float()

            output_y0, output_y1, output_z1, output_y0_c, output_y1_c = model(X_torch, z_torch, A_torch, V_act, V2_act, V_drop, V2_drop, noV)

            loss = np.zeros(nobs)
            for j in range(ny1):
                if loss_weights is None:
                    loss = loss + loss_function(output_y0[j], y0_torch[j]).detach().numpy()
                    loss = loss + loss_function(output_y1[j], y1_torch[j]).detach().numpy()
                else:
                    loss = loss + loss_function(output_y0[j], y0_torch[j]).detach().numpy() * loss_weights[0]
                    loss = loss + loss_function(output_y1[j], y1_torch[j]).detach().numpy() * loss_weights[1]
            if ny2 is not None:
                for j in range(ny2):
                    if loss_weights is None:
                        loss = loss + loss_function_c(output_y0_c[:,j], y0_c_torch[:,j]).detach().numpy()
                        loss = loss + loss_function_c(output_y1_c[:,j], y1_c_torch[:,j]).detach().numpy() 
                    else:
                        loss = loss + loss_function_c(output_y0_c[:,j], y0_c_torch[:,j]).detach().numpy() * loss_weights[2]
                        loss = loss + loss_function_c(output_y1_c[:,j], y1_c_torch[:,j]).detach().numpy() * loss_weights[3] 
            z_best[loss < min_loss,:] = z
            min_loss[loss < min_loss] = loss[loss < min_loss]

    res = {'z0': z_best, 'min_loss':min_loss}
    return res





###  predict best_z0 for all subjects in the test set --  minimal loss in Y0's only
def predict_z0 (run_net_object, dataset_t, n_test):
    
    net = run_net_object['model']

    nz = run_net_object['nz']
    ny1 = run_net_object['ny1']
    n_cat = run_net_object['n_cat']
    ny2 = run_net_object['ny2']  # added
    loss_weights = run_net_object['loss_weights']  # added on 0114
    
    min_loss = np.array([math.inf] * n_test)
    z_best = np.zeros(shape=(n_test, nz))
    
    loss_function = nn.NLLLoss(reduction='none')
    loss_function_c = nn.MSELoss(reduction='none')
        
    # prepare data
    dat_loader = DataLoader(dataset_t, batch_size=n_test, shuffle=False)
    for idx, (X_torch, y0_torch, z0_torch, y0_c_torch) in enumerate(dat_loader):

        for z in itertools.product([0,1], repeat=nz):   
            z_rep = np.array([z]*n_test)
            z_torch = torch.from_numpy(z_rep).float()

            output_y0 = []
            for j in range(ny1):
                output_y0.append(F.log_softmax(net.z_to_y[j](z_torch), dim=1))
            
            if ny2 is not None:   
                output_y0_c = net.z_to_y_c(z_torch)
                
            loss = np.zeros(n_test)  # for each subject
            for j in range(ny1):
                if loss_weights is not None:
                    loss = loss + loss_function(output_y0[j], y0_torch[j]).detach().numpy() * loss_weights[0]
                else:
                    loss = loss + loss_function(output_y0[j], y0_torch[j]).detach().numpy()
            if ny2 is not None:

                for j in range(ny2): 
                    if loss_weights is not None:
                        loss = loss + loss_function_c(output_y0_c[:,j], y0_c_torch[:,j]).detach().numpy() * loss_weights[2]  # for each subject
                    else:
                        loss = loss + loss_function_c(output_y0_c[:,j], y0_c_torch[:,j]).detach().numpy() # for each subject
    
            z_best[loss < min_loss,:] = z
            min_loss[loss < min_loss] = loss[loss < min_loss]
    
        accuracy_y0 = []
        for j in range(ny1):
            y0_temp = torch.argmax(output_y0[j], dim=1) 
            accuracy_y0.append(np.mean((y0_temp==y0_torch[j]).detach().numpy()))  
        accuracy_y0_all = np.mean(accuracy_y0)
        
        if ny2 is not None:
            mse_y0 = []
            for j in range(ny2):
                mse_y0.append(np.mean(loss_function_c(output_y0_c[:,j], y0_c_torch[:,j]).detach().numpy()))  # average over subjects
            mse_y0_all = np.mean(mse_y0)
        else:
            mse_y0 = None
            mse_y0_all = None

    res = {'z0': z_best, 'min_loss':min_loss, 'accuracy_y0':accuracy_y0, 'accuracy_y0_all':accuracy_y0_all, 'mse_y0':mse_y0, 'mse_y0_all':mse_y0_all}
    return res





#### Define function to find treatment that yields the lowest value in z1sum --- insample
def best_trt(run_net_object, nobs, dataset, z_weight=None, z_weight_cut=None):
    
    model = run_net_object['model']

    # define parameters:
    nz = run_net_object['nz']
    ny1 = run_net_object['ny1']
    n_cat = run_net_object['n_cat']
    
    nV = run_net_object['nV']
    V_act = run_net_object['V_act']
    nV2 = run_net_object['nV2']
    V2_act = run_net_object['V2_act']
    V_drop = run_net_object['V_drop']
    V2_drop = run_net_object['V2_drop']
    noV = run_net_object['noV']
    
    if z_weight is None:
        z_weight = np.array([1] * nz)
    
    dat_loader = DataLoader(dataset, batch_size=nobs, shuffle=False)
    for idx, (X_torch, z0_torch, A_torch, y0_torch, y1_torch, y0_c_torch, y1_c_torch) in enumerate(dat_loader):
    
        x_z0 = torch.cat((X_torch, z0_torch), 1)
 
        if noV is False:
            if V2_act is None:
                if V_act is "relu":
                    v_layer_act = F.relu(model.V(x_z0))
                elif V_act is "sigmoid":
                    v_layer_act = torch.sigmoid(model.V(x_z0))
            else:
                if V_act is "relu":
                    V2 = F.relu(model.V2(x_z0))
                    v_layer_act = F.relu(model.V(V2))
                elif V_act is "sigmoid":
                    V2 = torch.sigmoid(model.V2(x_z0))
                    v_layer_act = torch.sigmoid(model.V(V2))
             
            z1_layer_m1 = model.z1_m1(v_layer_act)
            z1_prob_m1 = torch.sigmoid(z1_layer_m1).detach().numpy()

            z1_layer_m0 = model.z1_m0(v_layer_act)
            z1_prob_m0 = torch.sigmoid(z1_layer_m0).detach().numpy()
        
        else:
            z1_layer_m1 = model.z1_m1(x_z0)
            z1_prob_m1 = torch.sigmoid(z1_layer_m1).detach().numpy()

            z1_layer_m0 = model.z1_m0(x_z0)
            z1_prob_m0 = torch.sigmoid(z1_layer_m0).detach().numpy()

        if z_weight is True:
            z_weight = cal_z_weight(run_net_object)['z_weight']
            if z_weight_cut is not None:
                z_weight[z_weight < z_weight_cut & z_weight > -1*z_weight_cut] = 0
            
        value_1 = np.sum(z1_prob_m1 * z_weight, axis=1)
        value_0 = np.sum(z1_prob_m0 * z_weight, axis=1)

        best_trt = 2 * (value_1 < value_0) - 1

        best_value = np.copy(value_0)
        best_value[best_trt==1] = value_1[best_trt==1]

        if(np.sum(value_1 ==value_0) >0):
            print("!!! value_1 ==value_0: ", np.sum(value_1 ==value_0))
            best_trt[value_1 ==value_0] = 2 * np.random.binomial(1, 0.5, np.sum(value_1 ==value_0)) - 1
    
    return {'opt_trt': best_trt, 'opt_value':best_value, "z1_prob_m0":z1_prob_m0, "z1_prob_m1":z1_prob_m1, "z_weight":z_weight}





#### Define function to find treatment that yields the lowest value in z1sum --- on test set 
def predict_trt(run_net_object, dataset_t, n_test, z_weight=None):
    
    model = run_net_object['model']

    nz = run_net_object['nz']
    ny1 = run_net_object['ny1']
    n_cat = run_net_object['n_cat']
    
    nV = run_net_object['nV']
    V_act = run_net_object['V_act']
    nV2 = run_net_object['nV2']
    V2_act = run_net_object['V2_act']
    V_drop = run_net_object['V_drop']
    V2_drop = run_net_object['V2_drop']
    noV = run_net_object['noV']
    
    if z_weight is None:
        z_weight = np.array([1] * nz)
    
    dat_loader = DataLoader(dataset_t, batch_size=n_test, shuffle=False)
    for idx, (X_torch, y0_torch, z0_torch, y0_c_torch) in enumerate(dat_loader): 
    
        x_z0 = torch.cat((X_torch, z0_torch), 1)
 
        if noV is False:
            if V2_act is None:
                if V_act is "relu":
                    v_layer_act = F.relu(model.V(x_z0))
                elif V_act is "sigmoid":
                    v_layer_act = torch.sigmoid(model.V(x_z0))
            else:
                if V_act is "relu":
                    V2 = F.relu(model.V2(x_z0))
                    v_layer_act = F.relu(model.V(V2))
                elif V_act is "sigmoid":
                    V2 = torch.sigmoid(model.V2(x_z0))
                    v_layer_act = torch.sigmoid(model.V(V2))
                
            z1_layer_m1 = model.z1_m1(v_layer_act)
            z1_prob_m1 = torch.sigmoid(z1_layer_m1).detach().numpy()

            z1_layer_m0 = model.z1_m0(v_layer_act)
            z1_prob_m0 = torch.sigmoid(z1_layer_m0).detach().numpy()
        
        else:
            z1_layer_m1 = model.z1_m1(x_z0)
            z1_prob_m1 = torch.sigmoid(z1_layer_m1).detach().numpy()

            z1_layer_m0 = model.z1_m0(x_z0)
            z1_prob_m0 = torch.sigmoid(z1_layer_m0).detach().numpy()

        if z_weight is True:
            z_weight = cal_z_weight(run_net_object)['z_weight']

        value_1 = np.sum(z1_prob_m1 * z_weight, axis=1)
        value_0 = np.sum(z1_prob_m0 * z_weight, axis=1)

        best_trt = 2 * (value_1 < value_0) - 1

        best_value = np.copy(value_0)
        best_value[best_trt==1] = value_1[best_trt==1]
        
        if(np.sum(value_1 ==value_0) > 0):
            print("value_1 ==value_0: ", np.sum(value_1 ==value_0))
            best_trt[value_1 ==value_0] = 2 * np.random.binomial(1, 0.5, np.sum(value_1 ==value_0)) - 1
    
    return {'opt_trt_test': best_trt, 'opt_value_test':best_value, "z1_prob_m0_test":z1_prob_m0, "z1_prob_m1_test":z1_prob_m1, "z_weight": z_weight}





### function to make prediction on the test set -- predict z0 first, then predict optimal treatment 
def predict_test(run_net_object, dataset_t, n_test, z_weight=None):
    
    n_test = int(n_test)
    z0_test = predict_z0(run_net_object, dataset_t, n_test)
    
    dataset_t_c = copy.deepcopy(dataset_t)
    dataset_t_c.z0 = torch.from_numpy(z0_test['z0']).float()

    opt_trt = predict_trt(run_net_object, dataset_t_c, n_test, z_weight)
    
    return {'z0_test': z0_test['z0'], 
            'accuracy_y0_test':z0_test['accuracy_y0'], 'accuracy_y0_all_test':z0_test['accuracy_y0_all'], 'z_weight':z_weight, **opt_trt}
            
            
         
                   
#### run_net for certain iterations (niter_model times), then update once z0 for each training subject -- run this for "niter_whole" times
def run_iters(nz, ny1, n_cat, y01, y11, X, A, seed=1, z0=None, ny2=None, y02=None, y12=None, 
              nV=10, V_act="relu", nV2=5, V2_act="relu", V_drop=None, V2_drop=None, noV=False,
              niter_model=10, niter_whole=10,  batch_size=100, optimizer="adam", lrate=0.1, lr_decay=1e-6, 
              momentum=0.5, nesterov=True, weight_decay=0, loss_weights=None, validation_split = 0.2, 
              cal_loss=True, show_fig = True, print_model=True, pred_trt=True,  
              dataset_t=None, n_test=None, 
              print_param=False, init_weight=False, set_zero_grad=False,
              z_weight=None, cal_value=False):

    nobs = np.shape(X)[0]
    n_train = nobs * (1-validation_split)
    
    if z_weight is None:
        z_weight = np.array([1]*nz)
        
    if seed is not None:
        np.random.seed(seed)

    net_object = run_net(nz, ny1, n_cat, y01, y11, X, A, z0=z0, ny2=ny2, y02=y02, y12=y12, 
                         nV=nV, V_act=V_act, nV2=nV2, V2_act=V2_act, V_drop=V_drop, V2_drop=V2_drop, noV=noV,
                         niter=niter_model, batch_size=batch_size, 
                         optimizer=optimizer, lrate=lrate, lr_decay=lr_decay, momentum=momentum, weight_decay=weight_decay, loss_weights=loss_weights, 
                         validation_split = validation_split, cal_loss=cal_loss, show_fig=show_fig, print_model=print_model, 
                         init_weight=init_weight, set_zero_grad=set_zero_grad)  
    
    for q in range(niter_whole):
        
        z0_new = best_z0_all(net_object, nobs, net_object['ds_all'])['z0']

        net_object = run_next_net(z0_new, net_object, niter_model, cal_loss=cal_loss, show_fig=False, print_model=False)
    
    if show_fig is True: 
        # plot loss
        plt.plot(net_object['loss'])
        plt.plot(net_object['loss_t'])
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()    
        # plot accuracy
        plt.plot(net_object['accuracy'])
        plt.plot(net_object['accuracy_t'])
        plt.title('Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        if ny2 is not None:
            plt.plot(net_object['mse'])
            plt.plot(net_object['mse_t'])
            plt.title('MSE for continuous Ys')
            plt.ylabel('MSE for continuous Ys')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()   
    
    if pred_trt is True:  # on the training set
        opt_trt = best_trt(net_object, nobs, net_object['ds_all'], z_weight=z_weight)
        
    if print_param is True:
        for name, param in net_object['model'].named_parameters():
            if param.requires_grad:
                print(name, param.data)
                
    if dataset_t is not None and n_test is not None:
        opt_trt_test = predict_test(net_object, dataset_t, n_test, z_weight=z_weight)
        
        return {'net_object':net_object, 'z0':z0_new, **opt_trt, **opt_trt_test}
    else:
        return {'net_object':net_object, 'z0':z0_new, **opt_trt}
    
    
    
    
    
#### function to run "run_iters" across different training sets ---- use this function to run simulation (and compare with OWL)
# will simulate training data using "sim_graph_binaryZ0"; data input is not None, will use the same input training data
# needs to input test data: data_t, dataset_t ...
# can use one training data only by specifying "data"
# loss_weights: a list of four -- for Y0, Y1, Y0_c, Y1_c loss
# R_t, A_t, pi_t to calculate value for the test set 
def run_iters_rep(start, end, nobs, nz, ny1, n_cat, nx,
                  a=None, W=None, theta0=None, theta1=None, theta2=None, theta3=None, theta4=None, theta5=None, 
                  ny2=None, a2=None, W2=None, 
                  nV=10, V_act="relu", nV2=20, V2_act="relu", V_drop=None, V2_drop=None, noV=False,
                  niter_model=10, niter_whole=10,  batch_size=100, optimizer="adam", lrate=0.1, lr_decay=1e-6, 
                  momentum=0.5, nesterov=True, weight_decay=0, loss_weights=None, validation_split = 0.2, 
                  cal_loss=True, show_fig = True, print_model=False, pred_trt=True,  
                  data=None, dataset_t=None, n_test=None, data_t=None,   #added data_t to calculate value
                  print_param=False, init_weight=False, set_zero_grad=False,  z_weight=None, 
                  run_owls=False, kernel="linear", c=[0.01, 0.1, 1, 10, 100], gamma=None, 
                  run_ours=True):
    
    nrep = end - start   
    res_mat = np.zeros(shape=(nrep, 20))
        
    net_objects = []
    res_owls = []
    
    if data is None:
        data_input = None
        
    for i in range(start, end):
        print("***********",i,"***********")
        
        # simulate with the given seed
        if data_input is None:
            data = sim_graph_binaryZ0(i, nobs, ny1, nz, nx, n_cat, a, W, theta0, theta1, theta2, theta3, theta4, theta5, ny2, a2, W2)
            
        if run_ours is True:
            res = run_iters(nz, ny1, n_cat, data['y0_obs'], data['y1_obs'], data['X'], data['trt'], seed=i,
                        ny2=ny2, y02=data['y0_obs_c'], y12=data['y0_obs_c'],
                        nV=nV, V_act=V_act,  nV2=nV2, V2_act=V2_act, V_drop=V_drop, V2_drop=V2_drop, noV=noV,
                        niter_model=niter_model, niter_whole=niter_whole,  batch_size=batch_size, 
                        optimizer=optimizer, lrate=lrate, lr_decay=lr_decay, momentum=momentum, 
                        nesterov=nesterov, weight_decay=weight_decay, loss_weights=loss_weights,
                        validation_split=validation_split, 
                        cal_loss=cal_loss, show_fig=show_fig, 
                        print_model=print_model, pred_trt=pred_trt, dataset_t=dataset_t, n_test=n_test,
                        print_param=print_param, init_weight=init_weight, set_zero_grad=set_zero_grad, 
                        z_weight=z_weight)
        
        if run_owls is True:
            H_tr = np.column_stack([data['y0_obs'], data['y0_obs_c'], data['X']])
            H_t = np.column_stack([data_t['y0_obs'], data_t['y0_obs_c'], data_t['X']])
            R_tr = np.sum(data['y1_obs'],axis=1) + np.sum(data['y1_obs_c'],axis=1)
            # run owl
            res_owl = run_owl(H=H_tr, H_test=H_t, R=-R_tr, A=data['trt'], pi=0.5, 
                              kernel=kernel, c=c, nfold=3, gamma=gamma, 
                              R_test=data_t['z1sum'], A_test=data_t['trt'], pi_test=0.5) 
        
        
        
        
        if run_ours is True and run_owls is True:
            # main results
            res_mat[i-start, 0] = np.mean(res['opt_trt'] == data['opt_trt'])
            res_mat[i-start, 1] = np.mean(res_owl['opt_trt'] == data['opt_trt'])
            res_mat[i-start, 2] = np.mean(res['opt_trt_test'] == data_t['opt_trt'])
            res_mat[i-start, 3] = np.mean(res_owl['opt_trt_test'] == data_t['opt_trt'])

            res_mat[i-start, 4] = np.mean(data['z1sum'] * (res['opt_trt'] == data['trt'])/ 0.5) 
            res_mat[i-start, 5] = np.mean(data['z1sum'] * (res_owl['opt_trt'] == data['trt'])/ 0.5) 
            res_mat[i-start, 6] = np.mean(data_t['z1sum'] * (res['opt_trt_test'] == data_t['trt'])/ 0.5) 
            res_mat[i-start, 7] = np.mean(data_t['z1sum'] * (res_owl['opt_trt_test'] == data_t['trt'])/ 0.5) 

            res_mat[i-start, 8] = np.mean(data['p_z1sum'] * (res['opt_trt'] == data['trt'])/ 0.5)
            res_mat[i-start, 9] = np.mean(data['p_z1sum'] * (res_owl['opt_trt'] == data['trt'])/ 0.5)
            res_mat[i-start, 10] = np.mean(data_t['p_z1sum'] * (res['opt_trt_test'] == data_t['trt'])/ 0.5) 
            res_mat[i-start, 11] = np.mean(data_t['p_z1sum'] * (res_owl['opt_trt_test'] == data_t['trt'])/ 0.5) 

            res_mat[i-start, 13] = np.mean(res['z0'] == data['z0'])
            res_mat[i-start, 14] = np.mean(res['net_object']['accuracy_y0'])
            res_mat[i-start, 15] = np.mean(res['net_object']['accuracy_y1'])
            res_mat[i-start, 16] = np.mean(res['z0_test'] == data_t['z0'])
            res_mat[i-start, 17] = res['accuracy_y0_all_test']
            
            # added results -- for comparing some Y1 values (those positive ones) on the test set
            res_mat[i-start, 18] = np.mean(data_t['y1sum_sel'] * (res['opt_trt_test'] == data_t['trt'])/ 0.5) 
            res_mat[i-start, 19] = np.mean(data_t['y1sum_sel'] * (res_owl['opt_trt_test'] == data_t['trt'])/ 0.5) 

        elif run_owls is True:
            res_mat[i-start, 1] = np.mean(res_owl['opt_trt'] == data['opt_trt'])
            res_mat[i-start, 3] = np.mean(res_owl['opt_trt_test'] == data_t['opt_trt'])
            res_mat[i-start, 5] = np.mean(data['z1sum'] * (res_owl['opt_trt'] == data['trt'])/ 0.5) 
            res_mat[i-start, 7] = np.mean(data_t['z1sum'] * (res_owl['opt_trt_test'] == data_t['trt'])/ 0.5) 
            res_mat[i-start, 9] = np.mean(data['p_z1sum'] * (res_owl['opt_trt'] == data['trt'])/ 0.5)
            res_mat[i-start, 11] = np.mean(data_t['p_z1sum'] * (res_owl['opt_trt_test'] == data_t['trt'])/ 0.5) 
            res_mat[i-start, 19] = np.mean(data_t['y1sum_sel'] * (res_owl['opt_trt_test'] == data_t['trt'])/ 0.5) 
        
        elif run_ours is True: 
            # main results
            res_mat[i-start, 0] = np.mean(res['opt_trt'] == data['opt_trt'])
            res_mat[i-start, 2] = np.mean(res['opt_trt_test'] == data_t['opt_trt'])

            res_mat[i-start, 4] = np.mean(data['z1sum'] * (res['opt_trt'] == data['trt'])/ 0.5) 
            res_mat[i-start, 6] = np.mean(data_t['z1sum'] * (res['opt_trt_test'] == data_t['trt'])/ 0.5) 

            res_mat[i-start, 8] = np.mean(data['p_z1sum'] * (res['opt_trt'] == data['trt'])/ 0.5)
            res_mat[i-start, 10] = np.mean(data_t['p_z1sum'] * (res['opt_trt_test'] == data_t['trt'])/ 0.5) 

            # other results
            res_mat[i-start, 13] = np.mean(res['z0'] == data['z0'])
            res_mat[i-start, 14] = np.mean(res['net_object']['accuracy_y0'])  # output directly from net_object
            res_mat[i-start, 15] = np.mean(res['net_object']['accuracy_y1'])
            res_mat[i-start, 16] = np.mean(res['z0_test'] == data_t['z0'])
            res_mat[i-start, 17] = res['accuracy_y0_all_test']  # output directly from net_object  
            
            # added results -- for comparing some Y1 values (those positive ones) on the test set
            res_mat[i-start, 18] = np.mean(data_t['y1sum_sel'] * (res['opt_trt_test'] == data_t['trt'])/ 0.5) 
        
        print(res_mat[i-start, ])
        
        if run_ours is True:
            net_objects.append(res['net_object'])
        if run_owls is True:
            res_owls.append(res_owl)
            
    if run_ours is True and run_owls is True :     
        return {'res':res_mat, 'net_objects': net_objects, 'res_owls':res_owls}          
    elif run_owls is True:
        return {'res':res_mat, 'res_owls':res_owls}          


    

####  create/simulate the data 
# a and W are list, one element for each categorical Y
# a2 and W2 are vectors, one entry for each continuous Y
def sim_graph_binaryZ0 (seed, nobs, ny1, nz, nx, n_cat, a, W, theta0, theta1, theta2, theta3, theta4, theta5, ny2=None, a2=None, W2=None, z0_p=0.5):
  
    np.random.seed(seed)

    z0 = np.random.binomial(n=1, p=z0_p, size=(nobs, nz))
    y0 = np.zeros(shape=(nobs, ny1), dtype='int8')

    for j in range(0, ny1):
        a_Wz0 = a[j] + z0 @ W[j]
        prob_ind = np.append(np.array([1] * nobs).reshape(nobs,1) , np.exp(a_Wz0), 1)
        prob = prob_ind / np.sum(prob_ind, axis=1)[:, None]
        for i in range(0, nobs):
            y0[i,j] = np.argmax( np.random.multinomial(1, prob[i,:]) )          
    
    if ny2 is not None and a2 is not None and W2 is not None:
        y0_c = a2 + z0 @ W2
    
    X = np.random.normal(size=(nobs, nx))
    trt = 2 * np.random.binomial(1, 0.5, size=nobs) - 1 
    
    theta_linear = theta0 + z0 @ theta1 + X @ theta2 + trt.reshape((nobs,1)) @ theta3.reshape(1,nz) + (np.array([trt] * nz).transpose() * z0) @ theta4 + (np.array([trt] * nx).transpose() * X) @ theta5
    p_z1 = np.exp(theta_linear) / (1 + np.exp(theta_linear))
    p_z1_sum = np.sum(p_z1, axis=1)
    z1 = np.random.binomial(n=1, p=p_z1)
  
    y1 = np.zeros(shape=(nobs, ny1), dtype='int8')
    for j in range(0, ny1):
        a_Wz1 = a[j] + z1 @ W[j]
        prob_ind_y1 = np.append(np.array([1] * nobs).reshape(nobs,1) , np.exp(a_Wz1), 1)
        prob_y1 = prob_ind_y1 / np.sum(prob_ind_y1, axis=1)[:, None]
        for i in range(0, nobs):
            y1[i,j] = np.argmax( np.random.multinomial(1, prob_y1[i,:]) )
     
    if ny2 is not None and a2 is not None and W2 is not None:
        y1_c = a2 + z1 @ W2 
        y0_c_sum = np.sum(y0_c, axis=1)
        y1_c_sum = np.sum(y1_c, axis=1)  

    y0_sum = np.sum(y0, axis=1)
    y1_sum = np.sum(y1, axis=1)
    z0_sum = np.sum(z0, axis=1)
    z1_sum = np.sum(z1, axis=1)

    trt_c = trt * (-1)
    theta_linear_c = theta0 + z0 @ theta1 + X @ theta2 + trt_c.reshape((nobs,1)) @ theta3.reshape(1,nz) + (np.array([trt_c] * nz).transpose() * z0) @ theta4 + (np.array([trt_c] * nx).transpose() * X) @ theta5
    p_z1_c = np.exp(theta_linear_c) / (1 + np.exp(theta_linear_c))
    p_z1_c_sum = np.sum(p_z1_c, axis=1)

    trt_better = p_z1_sum < p_z1_c_sum
    trt_c_better = p_z1_c_sum < p_z1_sum
    z1sum_eq = p_z1_sum == p_z1_c_sum
    z1sum_eq_n = sum(z1sum_eq) 

    opt_trt = 2 * np.random.binomial(n=1, p=0.5, size=(nobs)) - 1
    opt_trt[trt_better] = trt[trt_better]
    opt_trt[trt_c_better] = trt_c[trt_c_better]

    opt_trt2 = np.copy(opt_trt)
    opt_trt2[z1sum_eq] = 0

    opt_z1sum = p_z1_sum * trt_better + p_z1_c_sum * trt_c_better +  p_z1_sum * z1sum_eq
  
    z1sum_both = np.empty(shape=(nobs, 2), dtype='int8')
    z1sum_both[:,0] = p_z1_sum * (trt==1) + p_z1_c_sum * (trt_c==1)
    z1sum_both[:,1] = p_z1_sum * (trt==-1) + p_z1_c_sum * (trt_c==-1)

    trt_linear = np.array([theta3] * nobs) + z0 @ theta4 + X @ theta5
    opt_trts = 2 * (trt_linear < 0) - 1
    
    if ny2 is not None and a2 is not None and W2 is not None:
        y0_sum_all = y0_sum + y0_c_sum
        y1_sum_all = y1_sum + y1_c_sum
    else:
        y0_sum_all = y0_sum
        y1_sum_all = y1_sum
        y0_c = None
        y1_c = None
        
    res = {'nobs': nobs, 'X':X, 'trt':trt, 'y0_obs':y0, 'y1_obs':y1, 'z0':z0, 'z0_p':z0_p, 'z1':z1, 'y0_obs_c':y0_c, 'y1_obs_c':y1_c, 'y0sum':y0_sum, 'y1sum':y1_sum, 'y0sum_all':y0_sum_all, 'y1sum_all':y1_sum_all,
           'z0sum':z0_sum, 'z1sum':z1_sum, 'p_z1sum':p_z1_sum,
           'a':a, 'W':W, 'theta0':theta0, 'theta1':theta1, 'theta2':theta2, 'theta3':theta3, 'theta4':theta4, 'theta5':theta5, 'a2':a2, 'W2':W2, 
           'opt_trts':opt_trts, 'opt_trt':opt_trt, 'opt_trt2':opt_trt2, 'opt_z1sum':opt_z1sum, 'z1sum_both':z1sum_both, 'z1sum_eq_n':z1sum_eq_n, 'z1_p':p_z1, 'seed':seed}
    return res




# function to initialize weight, for the simulation 
def init_weights(net):
    for name, param in net.named_parameters():
        if name == "z_to_y.0.weight":
            param.data = torch.from_numpy(np.array([[0, 0, 0], [5, 0, 0]])).float()
        if name == "z_to_y.5.weight":
            param.data = torch.from_numpy(np.array([[0,0,0], [0,5,0], [0,10,0]])).float()
        if name == "z_to_y.6.weight":
            param.data = torch.from_numpy(np.array([[0,0,0], [0,0,5], [0,0,8], [0,0,10]])).float()
    return net