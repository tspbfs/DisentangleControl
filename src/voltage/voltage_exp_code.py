
import control
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import time


from plots import *

import torch
import torch.nn as nn
import torch.nn.functional as FF
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



import os
import sys
import pdb
import nolds
import copy



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def get_configs():

    parser = argparse.ArgumentParser(
        description='DISC Controller')
    parser.add_argument('--mode', default='Voltage', type=str,
                        help='Task Name')
    parser.add_argument('--ini_lambda', default=0.3, type=float,
                        help='initial trust parameter lambda, in [0,1]')
    parser.add_argument('--plot_curve', default=True,
                        type=bool, help='Plot tracking curve; only for mode = "Tracking"')
    parser.add_argument('--T', default=200,
                        type=int, help='Number of time slots')
    parser.add_argument('--scale', default=1.25,
                        type=float, help='scale of gaussian noise')    
    parser.add_argument('--amp', default=1,
                        type=float, help='amplify the total mixed noise')    
    parser.add_argument('--beta', default=0.08,
                        type=float, help='learning rate of the gradient')    
    parser.add_argument('--Buffer', default=50,
                        type=int, help='length of the buffer data at the beginning')    
    parser.add_argument('--gradstep', default=50,
                        type=int, help='steps for the computing of the gradient')        
    parser.add_argument('--fwdlength', default=10,
                        type=int, help='steps for the forward prediction')        
    parser.add_argument('--prefix', default='1',
                        type=int, help='index prefix for different runs')            

    configs = parser.parse_args()
    return configs


configs = get_configs()
print(configs)


directory = "OUTPUT/" + str(configs.prefix) + "_voltage_amp_" + str(configs.amp) + "_scale_" + str(configs.scale) + "_T_" + str(configs.T) 
directory += "_Beta_" + str(configs.beta) + "_Buffer_" + str(configs.Buffer) + "_Gradstep_" + str(configs.gradstep)


if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")






class Transcript(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):

        pass

sys.stdout = Transcript(directory + '/output.txt') 



print(device)




# Initialize

mode = 'Voltage'

T = configs.T



Node_num = 10

A = np.zeros((Node_num,Node_num))
B = np.array( [[2,2,2,2,2,2,2,2,2,2],
[2,4,4,4,4,4,4,4,4,4],
[2,4,8,4,8,8,4,4,8,8],
[2,4,4,8,4,4,8,8,4,4],
[2,4,8,4,12,8,4,4,12,12],
[2,4,8,4,8,12,4,4,8,8],
[2,4,4,8,4,4,12,8,4,4],
[2,4,4,8,4,4,8,12,4,4],
[2,4,8,4,12,8,4,4,20,12],
[2,4,8,4,12,8,4,4,12,20]
])


Q = np.identity(Node_num)
R = 0.1 * np.identity(Node_num)

P, _, _ = control.dare(A, B, Q, R)



print(A)
print(B)
print(P)

print(A.shape)
print(B.shape)
print('T ', T)


n_points = 1000
time_steps = np.arange(0, n_points)


scale_noise = configs.scale



pv_comp  = (np.asarray(np.load('stage_change/stage_change_data/pv.npy',allow_pickle=True))).astype(float)
wd_comp  = (np.asarray(np.load('stage_change/stage_change_data/wd.npy',allow_pickle=True))).astype(float)
gauss_comp = (np.asarray(np.load('stage_change/stage_change_data/gauss.npy',allow_pickle=True))).astype(float)
gauss_comp_scale = gauss_comp / scale_noise
Dataset1 = pv_comp + wd_comp + gauss_comp_scale 


_AMP_FACTOR_ = configs.amp
Dataset1 = Dataset1 * _AMP_FACTOR_




len_step = 1000
_COVR_len = 5 
_TARG_len = 5
batch_size_ = 64
EST_W_EPOCHS = 500



_BUFFERLEN_ = configs.Buffer
_FORWARDLEN_ = configs.fwdlength 
_GRADCOMPSTEP_ = configs.gradstep
_BETA_ = configs.beta




D = np.matmul(np.linalg.inv(R + np.matmul(np.matmul(np.transpose(B), P), B)), np.transpose(B))
H = np.matmul(B, D)
F = A - np.matmul(H, np.matmul(P, A))

F_list = [np.linalg.matrix_power(F, i) for i in range(T + 1)]
ft_length = max(T, _GRADCOMPSTEP_)

FT_list = [np.linalg.matrix_power(F.T, i) for i in range(ft_length + 1)]





############################################################################################


def partition_dataset(n, labeled_num = 500):
  train_num = labeled_num
  indices = np.random.permutation(n)
  train_indices, val_indices = indices[:train_num], indices[train_num:]
  return train_indices, val_indices


class subDataset(Dataset):

    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
 
    def __len__(self):
        return len(self.Data)
 
    def __getitem__(self, index):
        data = torch.FloatTensor(self.Data[index].reshape(10*_COVR_len))
        label = torch.FloatTensor(self.Label[index].reshape(10*_TARG_len))
        return data, label

    

class Regress(nn.Module):
    def __init__(self, covr_len, targ_len):
        super().__init__()

        SizeParam = 80
        self.fc1 = nn.Linear(10*_COVR_len, SizeParam)
        self.fc11 = nn.Linear(SizeParam, SizeParam)
        self.fc12 = nn.Linear(SizeParam, SizeParam)        
        self.fc14 = nn.Linear(SizeParam, SizeParam)
        self.fc2 = nn.Linear(SizeParam, 10*_TARG_len)
        
        
    def forward(self, x):              

        x1 = FF.leaky_relu(self.fc1(x))
        x1 = FF.leaky_relu(self.fc11(x1))
        x1 = FF.leaky_relu(self.fc12(x1))        
        x1 = FF.leaky_relu(self.fc14(x1))

        output = self.fc2(x1)
        return x1, output
    
def create_dataloader(dataset, len_step, covr_len, targ_len, offset = 750):
    covr = []
    targ = []

    for i in range(len_step - covr_len - targ_len- offset):
        covr.append(dataset[i:i+covr_len])
        targ.append(dataset[i+covr_len:i+covr_len+targ_len])

    covr_np = np.asarray(covr)
    covr_np = covr_np.reshape(len(covr_np),-1)
    targ_np = np.asarray(targ)
    targ_np = targ_np.reshape(len(targ_np),-1)
    
    
    _X = covr_np
    _Y = targ_np

    tr_indices, ts_indices = partition_dataset(len(_X),  len(_X)//10*8)
    X_tr = _X[tr_indices]
    X_ts = _X[ts_indices]
    Y_tr = _Y[tr_indices]
    Y_ts = _Y[ts_indices]

    X_ds  = subDataset(_X, _Y)
    X_tr_ds = subDataset(X_tr, Y_tr)
    X_ts_ds = subDataset(X_ts, Y_ts)

    x_total_loader = DataLoader(X_ds, batch_size=batch_size_ ,shuffle = True)
    x_tr_loader    = DataLoader(X_tr_ds, batch_size=batch_size_ ,shuffle = True)
    x_test_loader  = DataLoader(X_ts_ds, batch_size=batch_size_, shuffle=False)
    
    return x_total_loader, x_tr_loader, x_test_loader

def training_dataset(tr_loader, ts_loader ,epochs):

    reg_net = Regress(_COVR_len, _TARG_len)
    reg_net.to(device)
    reg_params = reg_net.parameters()
    reg_opt = optim.Adam(reg_params, lr = 1e-3)
    reg_cri = nn.MSELoss()
    test_cri = nn.MSELoss()

    losses = []
    losses_test = []
    for epoch in range(epochs):
        loss = 0.0
        for i, data in enumerate(tr_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            reg_opt.zero_grad()
            rep, y_pred = reg_net(inputs)        
            reg_loss = reg_cri(y_pred,labels)
            total_loss = reg_loss
            total_loss.backward()
            reg_opt.step()
            loss += reg_loss.item()
            if i % 10 == 0 and i >0:    
                
                losses.append(loss)
                loss = 0.0
            
    return losses_test, reg_net







def get_prediction_mixed(w, tr_epochs,t, forw_covr = None, _forw_length = 0):
    
    len_step = len(w)
    covr_len, targ_len, offset = _COVR_len, _TARG_len, 0
    tt_loader, tr_loader, ts_loader = create_dataloader(w, len_step, covr_len, targ_len, offset)
    

    losses_test, reg_net = training_dataset(tr_loader,ts_loader,tr_epochs)
    
    
    pred_lst= []
    for i in range(len_step-covr_len+1): 
        cov = torch.FloatTensor(w[i:i+covr_len,:].reshape(10*covr_len))
        cov = cov.to(device)
        rep, pred = reg_net(cov)        
        if i != len_step-covr_len:
            pred_lst.append(pred.reshape(targ_len,10)[0,:].cpu().detach().numpy())
        else:
            for j in range(5):
                pred_lst.append(pred.reshape(targ_len,10)[j,:].cpu().detach().numpy())
                
    pred_lst_np = np.asarray(pred_lst)
    
                
    for i in range(t):
        cov = torch.FloatTensor(pred_lst_np[-_COVR_len:,:].reshape(10*covr_len))
        cov = cov.to(device)
        rep, pred = reg_net(cov) 
        
        pred = pred.reshape(targ_len,10)[0,:].cpu().detach().numpy()
        pred_lst_np = np.concatenate( (pred_lst_np, pred.reshape(1,10)) ,axis=0)

            
    return pred_lst_np
 








############################################################################################
################ 
############################################################################################


def ent(w):
    res = 0
    for i in range(w.shape[1]):
        res += nolds.sampen(w[:,i])
    return res


def disentanglement(mixed_w,num_comp):
    transformer_w = FastICA(n_components=num_comp,
         random_state=0,
         whiten='unit-variance')
    X_transformed_w = transformer_w.fit_transform(mixed_w)
    A_mix_w = transformer_w.mixing_
    disentangle_comps = []
    for i in range(num_comp):
        rec_comp = np.outer(X_transformed_w[:,i],A_mix_w[:,i])
        disentangle_comps.append(rec_comp)
    disentangle_comps.append( np.mean(mixed_w,axis=0)  ) 

    return disentangle_comps, A_mix_w, X_transformed_w 



def _find_lam(t, w, estimated_w, P, F, H, ini_lambda):

    prediction_perturbation = 0
    prediction_prediction = 0

    for s in range(t):
        left_1 = 0
        left_2 = 0
        right = 0
        for l in range(s, t):
            left_1 += np.matmul(np.matmul(np.transpose(estimated_w[l]), np.transpose(P)), F[l - s])
            left_2 += np.matmul(np.matmul(np.transpose(w[l]), np.transpose(P)), F[l - s])
            right += np.transpose(np.matmul(np.matmul(np.transpose(estimated_w[l]), np.transpose(P)), F[l - s]))
        prediction_prediction += np.matmul(left_1, np.matmul(H, right))
        prediction_perturbation += np.matmul(left_2, np.matmul(H, right))

    if prediction_prediction != 0:
        lam_optimal = prediction_perturbation / prediction_prediction
    else:
        lam_optimal = ini_lambda

    return lam_optimal






def find_lam_grad_disc(t, record_w, estimated_w,  P, FT, H, num_comp, grad_step, lam_cur):

    
    t_num_comp = num_comp + 1
    _beta_ = _BETA_ 
    startInd = max(0,t-grad_step)
    endInd = t
            

    ksi = 0
    grad_ksi = 0        
    
    for i in range(startInd, endInd):        
        estimated_w_np = np.asarray(estimated_w[i])
        estimate_w_current = lam_cur[0] * estimated_w_np[0,:,:]    
        for j in range(1,t_num_comp):
            estimate_w_current += lam_cur[j] * estimated_w_np[j,:,:]
        
        ksi += FT[endInd-i-1] @ P @ ( record_w[t+_BUFFERLEN_-1,:] - estimate_w_current[t+_BUFFERLEN_-5-1,:])

        
    for i in range(startInd, endInd):        
        estimated_w_np = np.asarray(estimated_w[i])
        theta_s_current = np.transpose(estimated_w_np, (1,2,0)) 
        grad_ksi -= FT[endInd-i-1] @ P @ theta_s_current[t+_BUFFERLEN_-5-1,:,:]
        
    grad_lambda = grad_ksi.T @ (2*H) @ (ksi)

    return 0.5*_beta_*grad_lambda











def run_online_control_task(T, A, B, Q, R,  mode, P, D, H, F, ini_lambda, plot_curve, num_comp, FT):

    # Initialize
    _optimal_x = np.zeros((T, np.shape(A)[0]))
    OPT = 0
    
    _disc_x = np.zeros((T, np.shape(A)[0]))        
    disc_ALG = 0

    _selftune_x = np.zeros((T, np.shape(A)[0]))
    selftune_ALG = 0
    
    lqr_x = np.zeros((T, np.shape(A)[0]))
    lqr_CST  =  0 
    
    fulltrust_x = np.zeros((T, np.shape(A)[0]))           
    fullt_CST = 0
    

    ''' Hyperparameters'''
    T0 = _BUFFERLEN_    
    ForwLen = _FORWARDLEN_        
    grad_compute_step = _GRADCOMPSTEP_    
    t_num_comp = num_comp + 1  

    
    true_use_lam_disc = np.zeros((t_num_comp,T))    
    record_ws_disc = []    
    record_estimate_ws_disc = [] 
    

    true_use_lam_selftune = []    
    record_estimate_w_selftune = []            
    
    for t in range(T):        
        print("t", t)        


        _FTL_all_lam_selftune = _find_lam(t, Dataset1[T0 : T0 + t , :],  record_estimate_w_selftune, P, F, H, ini_lambda)

        if _FTL_all_lam_selftune < 0: 
            _FTL_all_lam_selftune = 0
        elif _FTL_all_lam_selftune > 1:   
            _FTL_all_lam_selftune = 1
        
        true_use_lam_selftune.append(_FTL_all_lam_selftune)  

        ####        
                        
        disent_results, mixing_theta, signal_ = disentanglement(Dataset1[:T0+t,:], num_comp)                                
        dist_comps = disent_results[:-1]
        w_mean = disent_results[-1]    
                                
        ''' Ordering by Entropy '''        
        entropy_comps = np.asarray([ent(i) for i in dist_comps])
        rank_by_entropy = entropy_comps.argsort()
        dist_comps = [dist_comps[i] for i in rank_by_entropy] 


        dist_comps.append(np.asarray([w_mean]*(len(dist_comps[0])) ))
                
        
        estimate_comps = []        
        forw_pred_comps = []
        pred_zeta_step = max(ForwLen, grad_compute_step)

        for i in range(num_comp):
            estimate_wi = get_prediction_mixed(dist_comps[i][:T0+t,:], EST_W_EPOCHS, pred_zeta_step-5 ) 
            estimate_comps.append(estimate_wi)
            forw_pred_comps.append(estimate_wi[-pred_zeta_step:,:])
                        
        estimate_comps.append(np.asarray([w_mean]*( len(estimate_comps[0]  ) )))
        forw_pred_comps.append(np.asarray([w_mean]*( len(forw_pred_comps[0] ))))
        record_estimate_ws_disc.append(estimate_comps)                                            
        record_ws_disc = Dataset1[:t+T0,:]
        
        
        estimated_w_selftune = get_prediction_mixed(Dataset1[:t+T0,:] , EST_W_EPOCHS, ForwLen-5) 
        
        forw_prediction_selftune = estimated_w_selftune[-ForwLen:,:]
        record_estimate_w_selftune.append(forw_prediction_selftune[0,:])
        

        if t == 0:
            tau_t = [ini_lambda]*t_num_comp
        else:    
            lam_grad = find_lam_grad_disc(t, record_ws_disc, record_estimate_ws_disc, P, FT, H, num_comp,grad_compute_step, lam_t)                        
            tau_t = tau_t - lam_grad
        
        lam_t = list(tau_t)

        #  Project  #
        for i in range(t_num_comp):
            if lam_t[i] < 0:
                lam_t[i] = 0
            if lam_t[i] > 1:
                lam_t[i] = 1   

        for i in range(t_num_comp):
            true_use_lam_disc[i][t] = lam_t[i]

        _FTL_all_lam_lst_disc = lam_t



        
        # Update actions

        # Disc algorithm 
        _disc_E = np.matmul(P, np.matmul(A, _disc_x[t]))                
        _disc_G_lst = [0]*t_num_comp        
        
        forw_step = min(ForwLen, T-t)
        
        for s in range(0, forw_step):                          
            for i in range(t_num_comp):                
                _disc_G_lst[i] += np.matmul(np.transpose(F[s]), np.matmul(P, forw_pred_comps[i][s]))         

        _disc_u = -np.matmul(D, _disc_E)         
        for i in range(t_num_comp):
            _disc_u -= _FTL_all_lam_lst_disc[i] * np.matmul(D, _disc_G_lst[i])\

            
        # Online Baseline 
        _selftune_E = np.matmul(P, np.matmul(A, _selftune_x[t]))        
        lqr_E = np.matmul(P, np.matmul(A, lqr_x[t]))
        fulltrust_E = np.matmul(P, np.matmul(A, fulltrust_x[t]))
        
        _myopic_G = 0
        
        for s in range(0, forw_step):            
            _myopic_G += np.matmul(np.transpose(F[s]), np.matmul(P, forw_prediction_selftune[s]))
                
        _selftune_u = -np.matmul(D, _selftune_E) - _FTL_all_lam_selftune * np.matmul(D, _myopic_G)        
        lqr_u = -np.matmul(D, lqr_E) - 0 * np.matmul(D, _myopic_G)
        fulltrust_u = -np.matmul(D, fulltrust_E) - 1 * np.matmul(D, _myopic_G)

        

            

        # Opt 
        _optimal_E = np.matmul(P, np.matmul(A, _optimal_x[t]))
        _optimal_G = 0
        
        for s in range(t+T0, T0+T):
            _optimal_G += np.matmul(np.transpose(F[s - t - T0 ]), np.matmul(P, Dataset1[s]))                        
        _optimal_u = -np.matmul(D, _optimal_E) - np.matmul(D, _optimal_G)


        # Update states
        if t < T - 1:
            _disc_x[t + 1] = np.matmul(A, _disc_x[t]) + np.matmul(B, _disc_u) + Dataset1[t+T0,:] 
            _optimal_x[t + 1] = np.matmul(A, _optimal_x[t]) + np.matmul(B, _optimal_u) + Dataset1[t+T0,:]
            _selftune_x[t + 1] = np.matmul(A, _selftune_x[t]) + np.matmul(B, _selftune_u) + Dataset1[t+T0,:]
            lqr_x[t + 1] = np.matmul(A, lqr_x[t]) + np.matmul(B, lqr_u) + Dataset1[t+T0,:]
            fulltrust_x[t + 1] = np.matmul(A, fulltrust_x[t]) + np.matmul(B, fulltrust_u) + Dataset1[t+T0,:]            

        
        # Update costs
        if (t < T - 1): 
            disc_ALG += np.matmul(np.transpose(_disc_x[t]), np.matmul(Q, _disc_x[t])) + np.matmul(
                np.transpose(_disc_u), np.matmul(R, _disc_u))            
            OPT += np.matmul(np.transpose(_optimal_x[t]), np.matmul(Q, _optimal_x[t])) + np.matmul(
                np.transpose(_optimal_u), np.matmul(R, _optimal_u))
            selftune_ALG += np.matmul(np.transpose(_selftune_x[t]), np.matmul(Q, _selftune_x[t])) + np.matmul(
                np.transpose(_selftune_u), np.matmul(R, _selftune_u))
            lqr_CST += np.matmul(np.transpose(lqr_x[t]), np.matmul(Q, lqr_x[t])) + np.matmul(
                np.transpose(lqr_u), np.matmul(R, lqr_u))
            fullt_CST += np.matmul(np.transpose(fulltrust_x[t]), np.matmul(Q, fulltrust_x[t])) + np.matmul(
                np.transpose(fulltrust_u), np.matmul(R, fulltrust_u))                                    
        
            
        elif t == T-1:
            disc_ALG += np.matmul(np.transpose(_disc_x[t]), np.matmul(P, _disc_x[t]))
            OPT += np.matmul(np.transpose(_optimal_x[t]), np.matmul(P, _optimal_x[t]))
            selftune_ALG += np.matmul(np.transpose(_selftune_x[t]), np.matmul(P, _selftune_x[t]))
            lqr_CST += np.matmul(np.transpose(lqr_x[t]), np.matmul(P, lqr_x[t]))
            fullt_CST += np.matmul(np.transpose(fulltrust_x[t]), np.matmul(P, fulltrust_x[t]))

    if plot_curve is True:

        np.save(directory+'/disc_lambda.npy',true_use_lam_disc)

        for i in range(t_num_comp):
            plt.figure()            
            plot_lambda(true_use_lam_disc[i,:])            
            plt.savefig(directory + '/lam_disc_' + str(i) + '.png')
            

    if plot_curve is True:

        np.save(directory+'/selftune_lambda.npy',true_use_lam_selftune)
        
        plt.figure()   
        plot_lambda(true_use_lam_selftune)        
        plt.savefig(directory + '/lam_selftune.png')
    

    print("Disentangled Online Cost is")
    print(disc_ALG)
    print("Optimal Cost is")
    print(OPT)

    print("selftune Cost is")
    print(selftune_ALG)

    print("LQR Cost is") 
    print(lqr_CST)
    print("fulltrust Cost is") 
    print(fullt_CST)

    

    return  disc_ALG, OPT








# Run online control
num_comp = 3

_disc_ALG, _OPT = run_online_control_task(T, A, B, Q, R,  mode, P,
                    D, H, F_list, configs.ini_lambda, configs.plot_curve, num_comp, FT_list)
        
        
                


print('\n\n\n\n\n\n')







def stop():
    """stop transcript and return print function to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal

stop()

















