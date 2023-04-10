from builtins import object, zip
import sys
import numpy as np
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append('./dependRepo/')
#from PiSL.ado import TrainSTRidge
from model import *
import os
from pathlib import Path


sys.path.append('../../../')

from subspace_inference import losses, utils
from datasetCollection import *
import pdb


from utilsODE import *
from spline import splineBasis
from ado import *
from sklearn.preprocessing import PolynomialFeatures
## set the fix seed
import scipy.io as sio
from scipy import sparse
##

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
import random
random.seed(1)
# make data
device = 'cuda:0'
print('using device:', device)

T_scale = 1


caseId = 'burgers'
idnum = 3
#caseId = 'KS'
noise_lv = 0.1
plotdir = 'figs/Spline_'+caseId+'New_Tscale'+str(T_scale)+'noise'+str(100*noise_lv)+'/'
savedir = 'savept/Spline_'+caseId+'New_Tscale'+str(T_scale)+'noise'+str(100*noise_lv)+'/'
print('plotdir is', plotdir)
Path(plotdir).mkdir(parents= True,exist_ok = True)
Path(savedir).mkdir(parents=True, exist_ok = True)

## replace this part with a function

polynomial_library = ['u_x', 'u*u_x', 'u**2*u_x', 'u**3*u_x',
                         'u_xx', 'u*u_xx', 'u**2*u_xx', 'u**3*u_xx',
                         'u_xxx', 'u*u_xxx', 'u**2*u_xxx']
dyna_lib = []

##

num_term = len(polynomial_library)
print('num_term is', num_term)
# create a function 
function_x1 = ''

for i in range(num_term):
    term = polynomial_library[i]
    ## shrink 
    #pdb.set_trace()
    #if term in relelib[0][0]:
    function_x1 += ('+cx1'+str(i)+'*'+term)
#pdb.set_trace()
function_x1 = function_x1[1:]

## load the dataset
dataloc = '../Data/'
splineloc = '../Data/spline2d/'
## burgers
caseName = 'burgers'
pre = caseName
post = 'Nm'
data = np.load(dataloc+'det'+str(idnum)+'txu.npz')
usol = data['u']
t = data['t1d']
t = t.reshape(-1,1)
x = data['x1d']
x = x.reshape(1,-1)
nI = 2
t = t[0::nI,:]
x = x[:,0::nI]
usol = usol[0::nI,0::nI].T


noisyob = usol+noise_lv*np.std(usol)*np.random.randn(*usol.shape)
namelist = ['t', 'x']
varlist,freqlist,Nclist = [t,x],[1,1],[25,25]

p = 3
device = 'cuda:0'

Qft  = sparse.load_npz(splineloc+pre+'QftNm'+'_p'+str(p)+'_freq'+str(freqlist[0])+str(freqlist[1])+'.npz')
QftNc = sparse.load_npz(splineloc+pre+'Qft'+post+'_p'+str(p)+'_freq'+str(freqlist[0])+str(freqlist[1])+'.npz')
# 1st derivative
QftNcdt = sparse.load_npz(splineloc+pre+'Qftdt'+post+'_p'+str(p)+'_freq'+str(freqlist[0])+str(freqlist[1])+'.npz')
QftNcdx = sparse.load_npz(splineloc+pre+'Qftdx'+post+'_p'+str(p)+'_freq'+str(freqlist[0])+str(freqlist[1])+'.npz')

# 2nd derivative
QftNcdx2 = sparse.load_npz(splineloc+pre+'Qftdx2'+post+'_p'+str(p)+'_freq'+str(freqlist[0])+str(freqlist[1])+'.npz')

# 3rd derivative
QftNcdx3 = sparse.load_npz(splineloc+pre+'Qftdx3'+post+'_p'+str(p)+'_freq'+str(freqlist[0])+str(freqlist[1])+'.npz')


#pdb.set_trace()
N = torch.sparse_coo_tensor(torch.Tensor([Qft.row.tolist(), Qft.col.tolist()]),
                            torch.Tensor(Qft.data), torch.Size(Qft.shape)).to(device)
Nc = N.clone()

Nc_dt =  torch.sparse_coo_tensor(torch.Tensor([QftNcdt.row.tolist(), QftNcdt.col.tolist()]),
                            torch.Tensor(QftNcdt.data), torch.Size(QftNcdt.shape)).to(device)

Nc_dx = torch.sparse_coo_tensor(torch.Tensor([QftNcdx.row.tolist(), QftNcdx.col.tolist()]),
                            torch.Tensor(QftNcdx.data), torch.Size(QftNcdx.shape)).to(device)

Nc_dx2 = torch.sparse_coo_tensor(torch.Tensor([QftNcdx2.row.tolist(),QftNcdx2.col.tolist()]),
                            torch.Tensor(QftNcdx2.data), torch.Size(QftNcdx2.shape)).to(device)

Nc_dx3 = torch.sparse_coo_tensor(torch.Tensor([QftNcdx3.row.tolist(),QftNcdx3.col.tolist()]),
                            torch.Tensor(QftNcdx3.data), torch.Size(QftNcdx3.shape)).to(device)


## load the x 

t_c_len = QftNc.shape[0]


#measurement = torch.Tensor(np.vstack([x1_noise,x2_noise,x3_noise, x4_noise, x5_noise, x6_noise]).transpose()).to(device)
measurement = torch.Tensor(noisyob.flatten('F')).to(device)
measurement = measurement.view(-1,1)

### add a lernable noise
#lnoise = torch.autograd.Variable(torch.rand(*measurement.shape).to(device), requires_grad = True)
###
#pdb.set_trace()

## sample from uniform distribution
P = torch.autograd.Variable(torch.rand(Qft.shape[1], 1).to(device), requires_grad=True)
#pdb.set_trace()
# sample from uniform ditribution. Can we sample from normal dist?

for i in range(num_term):
    globals()['cx1'+str(i)] = torch.autograd.Variable(torch.rand(1).to(device), requires_grad = True)

nstate = 1
coef_lst = [globals()['cx1'+str(i)] for i in range(num_term)]
dyna_lib = polynomial_library*nstate

start_time = time.time()

# make loss
learning_rate = 1e-2
parameters = [{'params': [P]},
                {'params': coef_lst}]#,
                #{'params': [lnoise]}]
optimizer = torch.optim.Adamax(parameters,lr=learning_rate)

scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 200, min_lr = 0.00001)

## change loss function
loss_func = losses.SplineDataLikelihood(1)
loss_func1 = losses.SplineEqLikelihood(1)

epoch = 50000
#epoch = 30000

err_best = 10000
min_loss = 10000
epochs_no_improve = 0

E1_test = []
LOSS = LOSS1 = LOSS2 = LOSSEQ = []

swag_model = SplineSWAG([P]+coef_lst, subspace_type="pca", 
                    subspace_kwargs={"max_rank": 10, "pca_rank": 10})
swag_model_init = swag_model.state_dict().copy()
#swag_start= 50000

swag = True
Train_flag = False
postProcess = True
lr_init = learning_rate
swag_lr = learning_rate/10
ADO = True
#posttuning = True

true_x1 = '-u_x'
if Train_flag == True:
    print('use adamax for pretrain')
    for i in range(epoch):
        optimizer.zero_grad()
        loss = 0
        loss1, _,_ = loss_func(N, P, measurement, device = device)
        u = torch.matmul(Nc, P[:, 0])

        u_x = torch.matmul(Nc_dx, P[:,0])
        u_xx = torch.matmul(Nc_dx2, P[:,0])
        u_xxx = torch.matmul(Nc_dx3, P[:,0])

        rhs_x1 = eval(function_x1)
        loss2, _,_ = loss_func1(rhs_x1, Nc_dt, P, rhs_y = None, rhs_z = None, rhs_x4 = None, rhs_x5 = None, rhs_x6 = None)

        loss = loss1+1/T_scale*loss2
        ## and the penalty
        tmp = torch.Tensor(coef_lst)
        loss3 = 1e-7*torch.norm(tmp,p=1)
        loss += loss3
        ##
        loss.backward()

        scheduler.step(loss)
        optimizer.step()

        if loss.item() >= min_loss:
            epochs_no_improve += 1
        else:
            min_loss = loss.item()
            epochs_no_improve = 0
        if epochs_no_improve == 100 and optimizer.param_groups[0]['lr'] == 0.00001:
            print('epoch :',i, 'loss :', loss.item(), 'lrP :', optimizer.param_groups[0]['lr'], ' lrC', optimizer.param_groups[1]['lr'])
            print('Early stopping')
            loss_pretuning = loss.item()
            print("--- %s seconds ---" % (time.time() - start_time))
            break


        if i % 5000 == 0:
            #print('coeff is', coef_lst)
            LOSS.append(loss.item())
            LOSS1.append(loss1.item())
            LOSS2.append(loss2.item())
            #print('epoch is '+str(i)+' loss total is ', loss.item(), ' loss data is', (loss1+loss2).item())
            print('epoch is '+str(i)+' loss total is ', loss.item(), ' loss eq is', loss2.item(), ' loss data is', loss1.item(),'lr :', optimizer.param_groups[0]['lr']) 
        # stop training in the 100,000 epoch
        if i == 49999:
            print('epoch :', i, 'loss :', loss.item(), 'lr :', optimizer.param_groups[0]['lr'])
            print("50,000 epochs completed!")
            loss_pretuning = loss.item()
            print("--- %s seconds ---" % (time.time() - start_time)) 
    save_checkpoint(
                dir = savedir,
                epoch = epoch,
                name="swag_checkpoint",
                state_dict=swag_model.state_dict(),
                )
if postProcess == True:
    #from visualization import plot_predictive1

    #relevant_term = [6,9,16,19]
    #relevant_term = [1,2,21,22,26,45,43]
    swag_model = SplineSWAG([P]+coef_lst, subspace_type="pca", 
                    subspace_kwargs={"max_rank": 10, "pca_rank": 10})
    swag_model.load_state_dict(torch.load(savedir+"swag_checkpoint-"+str(epoch)+".pt")["state_dict"])


    trajectories = []
    sampleTraject = []
    #pdb.set_trace()
    Pmean = swag_model.state_dict()['param1'].clone()

## ADO part
if ADO == True:
    #loss_pretuning = loss.item()
    ## Psample better be changed by P
    P_pre = Pmean.cpu().detach().numpy().copy()
    lambda_raw = np.zeros([num_term, nstate])

    function_x1 = ''
    for i in range(0, num_term):
        term = polynomial_library[i]
        if globals()['cx1'+str(i)] in coef_lst:
            function_x1 += (' + '+str(np.round(globals()['cx1'+str(i)].cpu().detach().numpy()[0],3))+'*'+term)
            lambda_raw[i,0] = globals()['cx1'+str(i)].cpu().detach().numpy()[0]
    # pinrt pre-tuned equations
    print('u_t/=:', function_x1.replace('+ -', '- ')[3:] )
    print()

    loss_HY = []
    loss_HY_min = 1000000

    terms_HY = [num_term*nstate]
    A_raw = lambda_raw.copy()
    A_raw_HY = A_raw.copy()

    P_HY_np = P_pre.copy()
    P_HY = torch.autograd.Variable(torch.Tensor(P_HY_np).to(device), requires_grad = True)

    diminish_coef = True
    num_terms = np.count_nonzero(A_raw)
    ## ad hoc parameters
    tol = 0.1
    #tol = 0.025
    d_tol = 0.01
    #d_tol = 1
    lam = 1e-6 # ridge regularizer
    #eta = 0.001 # l-0 penalty ratio
    #lam = 1e-7
    #eta = 1e-4
    #eta = 0.01
    eta = 0.001
    tol_best = [0]
    start_time = time.time()

    itr = 0
    itr_max = 5
    while diminish_coef or itr < itr_max:
        ## ADO part1 : STRidge training

        print('itr' + str(itr+1))
        print('Training parameters (STRidge):')
        u = torch.matmul(Nc, P_HY[:, 0]).cpu().detach().numpy()
        u_x = torch.matmul(Nc_dx, P_HY[:,0]).cpu().detach().numpy()
        u_xx = torch.matmul(Nc_dx2, P_HY[:,0]).cpu().detach().numpy()
        u_xxx = torch.matmul(Nc_dx3, P_HY[:,0]).cpu().detach().numpy()

        phi = np.zeros([t_c_len, num_term])
        for i in range(num_term):
            # seems not useful to include replace
            phi[:, i] = eval(polynomial_library[i].replace('torch.', ''))
        Y_spline = torch.matmul(Nc_dt, P_HY).cpu().detach().numpy()
        A_raw[:, 0], tol_best[0] = TrainSTRidge(phi, Y_spline[:,0], lam, eta, d_tol, maxit = 500)

        print('best tolerance threhold is', tol_best)
        print('prune number of terms to', np.count_nonzero(A_raw))
        print()

        print('Spline training')
        function_x1 = ''
        sparse_c_lst = []
        for i in range(0, num_term):
            term = polynomial_library[i]
            if A_raw[i, 0]!= 0:
                function_x1 += (' + cx1'+str(i)+'*'+term)
                sparse_c_lst.append(globals()['cx1'+str(i)])
        function_x1 = function_x1[3:]
        learning_rate = 0.05
        parameters = [{'params': [P_HY]},
                {'params': sparse_c_lst}]#,
                #{'params': [lnoise]}]
        optimizer = torch.optim.Adam(parameters,lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 200, min_lr = 0.0001)

        err_best = 10000
        min_loss = 10000
        epochs_no_improve = 0

        loss_his = []

        for ki in range(20000):
            optimizer.zero_grad()
            loss1, _,_ = loss_func(N, P_HY, measurement, device = device)
            u = torch.matmul(Nc, P_HY[:, 0])
            u_x = torch.matmul(Nc_dx, P_HY[:,0])
            u_xx = torch.matmul(Nc_dx2, P_HY[:,0])
            u_xxx = torch.matmul(Nc_dx3, P_HY[:,0])

            rhs_x1 = eval(function_x1)
            loss2, _,_ = loss_func1(rhs_x1, Nc_dt, P_HY, rhs_y = None, rhs_z = None, rhs_x4 = None, rhs_x5 = None, rhs_x6 = None)
            loss = loss1+1/T_scale*loss2

            tmp = torch.Tensor(sparse_c_lst)
            loss3 = 1e-7*torch.norm(tmp,p=1)
            loss+=loss3
            loss.backward()

            scheduler.step(loss)

            optimizer.step()

            if loss.item() >= min_loss:
                epochs_no_improve += 1
            else:
                min_loss = loss.item()
                epochs_no_improve = 0
            
            if epochs_no_improve == 100 and optimizer.param_groups[0]['lr'] == 0.0001:
                print('epoch :', ki , 'loss :', loss.item(), 'lrP :', optimizer.param_groups[0]['lr'], ' lrC', optimizer.param_groups[1]['lr'])
                print('Early stopping')
                
                break
            
            if ki%5000 == 0:
                print('epoch :', ki, 'loss :', loss.item(), 'lrP :', optimizer.param_groups[0]['lr'], 'lrC', optimizer.param_groups[1]['lr'])
            optimizer.step()
            if ki == 19999:
                print('epoch :', ki, 'loss :', loss.item(), 'lrP :', optimizer.param_groups[0]['lr'], 'lrC', optimizer.param_groups[1]['lr'])
                print('20000 epochs completed')
        
        for i in range(num_term):
            if A_raw[i, 0]!= 0: A_raw[i, 0] = globals()['cx1'+str(i)].cpu().detach().numpy()[0]

        for i in range(A_raw.shape[0]):
            for j in range(A_raw.shape[1]):
                if abs(A_raw[i, j]) < tol:
                    A_raw[i, j] = 0

        print('prune number of terms to', np.count_nonzero(A_raw))
        ## this one needs to be modified
        loss_HY.append(loss.item() + eta*np.count_nonzero(A_raw))
        terms_HY.append(np.count_nonzero(A_raw))
        if loss_HY[-1] < loss_HY_min:
            A_raw_HY = A_raw.copy()
            loss_HY_min = loss_HY[-1]
        if np.count_nonzero(A_raw) < num_terms:
            num_terms = np.count_nonzero(A_raw)
        else:
            diminish_coef = False
        itr += 1
        print()
        function_x1 = ''
        for i in range(0, num_term):
            term = polynomial_library[i]
            if A_raw_HY[i,0] != 0: function_x1 += (' + '+str(np.round(A_raw_HY[i,0], 4))+'*'+term)
        print()
        print('u_t/=', function_x1[3:].replace('+ -', '- '))
    print('reach convergence of number of terms in governing equations!')
    print("--- %s seconds ---" % (time.time() - start_time))
    print()
    print('final result :')

    function_x1 = ''
    for i in range(0, num_term):
        term = polynomial_library[i]
        if A_raw_HY[i, 0] != 0: function_x1 += (' + '+str(np.round(A_raw_HY[i,0], 4))+'*'+term)
    print()
    print('u_t/=', function_x1[3:].replace('+ -', '- '))
    print()

    np.savez(caseId+'ADOCoeff',A_raw_HY = A_raw_HY.copy(), P_HY = P_HY.cpu().detach().numpy())

    save_checkpoint(
                dir = savedir,
                epoch = epoch,
                name="ADOswag_checkpoint",
                state_dict=swag_model.state_dict(),
                )
    nz = A_raw_HY[[1,4],:]
    truecoef = np.array([-1,0.5])
    truecoef = truecoef[:,None]
    rmse2 = np.linalg.norm(nz-truecoef)/np.linalg.norm(truecoef)
    print('coef rmse is', rmse2)