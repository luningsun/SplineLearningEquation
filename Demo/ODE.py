from builtins import object
import numpy as np
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append('./dependRepo/')
#from PiSL.ado import TrainSTRidge
from model import *
import os
from pathlib import Path

import scipy.io as sio
sys.path.append('../../../')

from subspace_inference import losses, utils
from datasetCollection import *
import pdb


from utilsODE import *
from spline import splineBasis
from ado import *
from sklearn.preprocessing import PolynomialFeatures
## set the fix seed

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
Dnoise_lv = 5
freq = 1
#categ = 'syntheticdata'
categ = 'realdata'
caseId = 'LotkaVolterra'
jp = 100
prefix = 'parfit_final'
#prefix = ''
if 'synthetic' in categ:
    plotdir = 'figs/'+prefix+'freq'+str(freq)+'jump'+str(jp)+categ+' Spline_'+caseId+'_Tscale'+str(T_scale)+'/'
    savedir = 'ckpts/'+prefix+'freq'+str(freq)+'jump'+str(jp)+categ+' Spline_'+caseId+'_Tscale'+str(T_scale)+'/'
else:
    plotdir = 'figs/'+prefix+'freq'+str(freq)+categ+' Spline_'+caseId+'_Tscale'+str(T_scale)+'/'
    savedir = 'ckpts/'+prefix+'freq'+str(freq)+categ+' Spline_'+caseId+'_Tscale'+str(T_scale)+'/'
print('plotdir is', plotdir)
Path(plotdir).mkdir(parents= True,exist_ok = True)
Path(savedir).mkdir(parents=True, exist_ok = True)
# ['toy','cubic', 'linear','const']

#caseId  = 'constant'

## option1 cubic spline
poly_degree = 2
nstate = 2
X = np.arange(poly_degree * nstate).reshape(poly_degree, nstate)
poly = PolynomialFeatures(poly_degree)
poly.fit_transform(X)
raw_poly = poly.get_feature_names(input_features=['x','y'])
polynomial_library = [p.replace(' ', '*').replace('^', '**') for p in raw_poly]
##remove costant
polynomial_library = polynomial_library[1:]
if 'parfit' in prefix:
    ## correct polynomial_library
    tlib = [0,1,3]
    polynomial_library = [polynomial_library[ttlib] for ttlib in tlib]
    #polynomial_library.append('1')
    polynomial_library.append('x**2*y')
    polynomial_library.append('x*y**2')
#pdb.set_trace()
##
dyna_lib = []


Xi_base = np.zeros([len(polynomial_library),2])
if 'parfit' in prefix:
    #Xi_base = np.zeros([3,2])
    Xi_base[0,0] = 0.4807
    Xi_base[2,0] = -0.0248
    Xi_base[1, 1] = -0.9272
    Xi_base[2,1] =  +0.0276
else:
    ## assign nonzero
    tmplst1 = [0,-2]
    tmplst2 = [1,-2]
    for idl1 in tmplst1:
        Xi_base[idl1,0] = 1
    for idl2 in tmplst2:
        Xi_base[idl2,1] = 1
    Xi_base[0,0] = 0.4807
    Xi_base[-2,0] = -0.0248
    Xi_base[1, 1] = -0.9272
    Xi_base[-2,1] =  +0.0276



#pdb.set_trace()
if 'synthetic' in categ:
    ### load the predator prey data
    tmpdata = sio.loadmat('../Data/lvdata.mat')
    ## define x,y as the sim data
    x = tmpdata['tsim'].T
    y = tmpdata['ysym']
    ## subsample to control 2000 points
    x = x[0::jp,:]
    y = y[0::jp,:]
else:
    ## realdatacase
    tmpdata = sio.loadmat('../Data/lvdata.mat')
    tnoise = tmpdata['td'].T
    x = tmpdata['td'].T
    y1 = tmpdata['hare'].T
    y2 = tmpdata['lynx'].T
    y = np.concatenate((y1,y2),1)
    ysim = tmpdata['ysym']
##
##
#pdb.set_trace()

####



init_cond = y[0,:].copy()
end_cond = y[-1,:].copy()
#pdb.set_trace()


##
end_t = int(np.max(x))
num_control = freq*end_t+1
num_c = 1000*end_t+1
# define the knots function
t = np.linspace(0, end_t, num_control)
#pdb.set_trace()
knots = np.array([0,0,0]+list(t) + [end_t,end_t,end_t])
#pdb.set_trace()
t_m_all = x.copy()
#sub_idx = range(0, len(x), 5)
if 'realdata' in categ:
    sub_idx = range(0,len(x))
else:
    sub_idx = range(0,len(x),10)
t_m = t_m_all[sub_idx]


##
x_noise = y[sub_idx,0].copy()
y_noise = y[sub_idx,1].copy()
#pdb.set_trace()
tplot = t_m.copy()
yplot = np.column_stack((x_noise,y_noise)).copy()
data_plot = np.concatenate((tplot,yplot),1)
#pdb.set_trace()
t_c = np.array(sorted(list(t_m) + list(np.random.rand(num_c-len(t_m))*end_t)),dtype = object)

basis = splineBasis(knots, t_m, t_c)
basis_m, basis_dt_m = basis.get_measurement()
basis_c, basis_dt_c = basis.get_collocation()
if 'synthetic' in categ:
    np.savez('../Data/freq'+str(freq)+'jump'+str(jp)+categ+caseId+'spline', basis_m = basis_m, basis_dt_m = basis_dt_m, basis_c = basis_c, basis_dt_c = basis_dt_c)
else:
    np.savez('../Data/freq'+str(freq)+categ+caseId+'spline', basis_m = basis_m, basis_dt_m = basis_dt_m, basis_c = basis_c, basis_dt_c = basis_dt_c)

if 'synthetic' in categ:
    data = np.load('../Data/freq'+str(freq)+'jump'+str(jp)+categ+caseId+'spline.npz')
else:
    data = np.load('../Data/freq'+str(freq)+categ+caseId+'spline.npz')

basis_m = data['basis_m']
basis_dt_m = data['basis_dt_m']
basis_c = data['basis_c']
basis_dt_c = data['basis_dt_c']
print('shape of basis_m is',basis_m.shape)
print('shaep of basis_dt_m is', basis_dt_m.shape)
print('shape of basis_c is', basis_c.shape)
print('shape of basis_dt_c is', basis_dt_c.shape)
#pdb.set_trace()


trueName = []
relevant_term = []
relevant_term2 = []
relelib = []
true_coef = []
for i in range(nstate):
    stateName = []
    
    idx = np.where(Xi_base[:,i]!=0)[0]
    

    #pdb.set_trace()
    stateName.append([polynomial_library[tt] for tt in idx])
    trueName.append(stateName)
    print('lib for state ',str(i+1),' is', stateName)
    print('coeff for state ', str(i+1), ' is', Xi_base[idx,i])
    print('\n')
    relevant_term2.append(idx)
    idx = idx+(i)*len(polynomial_library)
    relevant_term.append(idx)
    relelib.append(stateName)
    idx2 = np.where(Xi_base[:,i]!=0)[0]
    true_coef.append(Xi_base[idx2,i])

print('rele term is', relevant_term)
print('rele lib is', relelib)
print('true_coef is', true_coef)



##
#pdb.set_trace()

t_m_len = basis_m.shape[0]
t_c_len = basis_c.shape[0]
num_control = basis_m.shape[1]
num_term = len(polynomial_library)
print('num_term is', num_term)
# create a function 
function_x = function_y = ''

for i in range(num_term):
    term = polynomial_library[i]
    function_x += ('+cx'+str(i)+'*'+term)
    function_y += ('+cy'+str(i)+'*'+term)

function_x = function_x[1:]
function_y = function_y[1:]

N = torch.Tensor(basis_m).to(device)
N_c = torch.Tensor(basis_c).to(device)
N_dt = torch.Tensor(basis_dt_c).to(device)

measurement = torch.Tensor(np.vstack([x_noise,y_noise]).transpose()).to(device)

## sample from uniform distribution
P = torch.autograd.Variable(torch.rand(num_control,nstate).to(device), requires_grad=True)

# sample from uniform ditribution. Can we sample from normal dist?
for i in range(num_term):
    globals()['cx'+str(i)] = torch.autograd.Variable(torch.rand(1).to(device), requires_grad = True)
    globals()['cy'+str(i)] = torch.autograd.Variable(torch.rand(1).to(device), requires_grad = True)

coef_lst = [globals()['cx'+str(i)] for i in range(num_term)] + \
            [globals()['cy'+str(i)] for i in range(num_term)]
dyna_lib = polynomial_library*nstate

#pdb.set_trace()
start_time = time.time()


# make loss
learning_rate = 1e-2
parameters = [{'params': [P]},
                {'params': coef_lst}]

optimizer = torch.optim.Adamax(parameters,lr=learning_rate)
print('use adamax for pretrain')
scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 200, min_lr = 0.00001)


## change loss function
loss_func = losses.SplineDataLikelihood(0.5)
loss_func1 = losses.SplineEqLikelihood(0.5)
#pdb.set_trace()

epoch = 851000

err_best = 10000
min_loss = 10000
epochs_no_improve = 0
#pdb.set_trace()
#p=[-0.1,2,-2,-0.1]
#p= np.array([[-0.1,2],[-2,-0.1]])#/100
p = np.array([0.5])
#pdb.set_trace()

E1_test = E2_test = []
LOSS = LOSS1 = LOSS2 = LOSSEQ = []

swag_model = SplineSWAG([P]+coef_lst, subspace_type="pca", 
                    subspace_kwargs={"max_rank": 10, "pca_rank": 10})
swag_model_init = swag_model.state_dict().copy()
#swag_start= 50000
#swag = False
Train_flag = False
postProcess = True
lr_init = learning_rate
swag_lr = learning_rate/10
ADO = True
posttuning = True

no_improve_val = 10000
#pdb.set_trace()
if Train_flag == True:
    for i in range(epoch):
        optimizer.zero_grad()
        loss = 0
        loss1,_,_, = loss_func(N,P,measurement)

        x = torch.matmul(N_c, P[:,0])
        y = torch.matmul(N_c, P[:,1])

        rhs_x = eval(function_x)
        rhs_y = eval(function_y)
        loss2,_,_ = loss_func1(rhs_x, N_dt, P, rhs_y = rhs_y)
        loss = loss1+1/T_scale*loss2
        loss.backward()

        scheduler.step(loss)
        optimizer.step()

        if loss.item() >= min_loss:
            epochs_no_improve += 1
        else:
            min_loss = loss.item()
            epochs_no_improve = 0
        if epochs_no_improve == no_improve_val and optimizer.param_groups[0]['lr'] == 0.00001:
            print('epoch :',i, 'loss :', loss.item(), 'lrP :', optimizer.param_groups[0]['lr'], ' lrC', optimizer.param_groups[1]['lr'])
            print('Early stopping')
            loss_pretuning = loss.item()
            print("--- %s seconds ---" % (time.time() - start_time))
            break
        if i % 5000 == 0:
            LOSS.append(loss.item())
            LOSS1.append(loss1.item())
            LOSS2.append(loss2.item())
            print('epoch is '+str(i)+' loss total is ', loss.item(), ' loss eq is', loss2.item(), ' loss data is', loss1.item(),'lr :', optimizer.param_groups[0]['lr'])
        if i == epoch:
            print('epoch :', i, 'loss :', loss.item(), 'lr :', optimizer.param_groups[0]['lr'])
            print("pretrain completed!")
            loss_pretuning = loss.item()
            print("--- %s seconds ---" % (time.time() - start_time))                            

    save_checkpoint(
                dir = savedir,
                epoch = epoch,
                name="swag_checkpoint",
                state_dict=swag_model.state_dict(),
                )
print("--- %s seconds ---" % (time.time() - start_time))  
pdb.set_trace()
#relevant_term = np.hstack(np.array(relevant_term))
if postProcess == True:
    #from visualization import plot_predictive1

    #relevant_term = [6,9,16,19]
    
    swag_model = swag_model = SplineSWAG([P]+coef_lst, subspace_type="pca", 
                    subspace_kwargs={"max_rank": 10, "pca_rank": 10})
    swag_model.load_state_dict(torch.load(savedir+"swag_checkpoint-"+str(epoch)+".pt")["state_dict"])

    #data1 = np.load('collectors.npz')
    #w_collector = data1['w_collector']
    #mean_collector = data1['mean_collector']

    trajectories = []
    sampleTraject = []
    #pdb.set_trace()
    Pmean = swag_model.state_dict()['param1'].clone()
    
    relevant_term = [relevant_term[k].tolist() for k in range(nstate)]
    relevant_term = relevant_term[0]+relevant_term[1]
    dyna_lib_rele = list(dyna_lib[k] for k in relevant_term)


## ADO part
if ADO == True:
    #pdb.set_trace()
    #loss_pretuning = loss.item()
    ## Psample better be changed by P
    P_pre = Pmean.cpu().detach().numpy().copy()
    #P_pre = Psample.cpu().detach().numpy().copy()

    lambda_raw = np.zeros([num_term, nstate])

    function_x = function_y = ''

    for i in range(0, num_term):
        term = polynomial_library[i]
        if globals()['cx'+str(i)] in coef_lst:
            function_x += (' + '+str(np.round(globals()['cx'+str(i)].cpu().detach().numpy()[0],3))+'*'+term)
            lambda_raw[i,0] = globals()['cx'+str(i)].cpu().detach().numpy()[0]
        if globals()['cy'+str(i)] in coef_lst:
            function_y += (' + '+str(np.round(globals()['cy'+str(i)].cpu().detach().numpy()[0],3))+'*'+term)
            lambda_raw[i,1] = globals()['cy'+str(i)].cpu().detach().numpy()[0]
        
    # pinrt pre-tuned equations
    print('x_dot :', function_x.replace('+ -', '- ')[3:] )
    print()
    print('y_dot :', function_y.replace('+ -', '- ')[3:] )



    
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
    tol = 0.01
    d_tol = 0.01
    lam = 1e-6 # ridge regularizer
    #eta = 0.001 # l-0 penalty ratio
    eta = 0.005

    tol_best = [0, 0]

    start_time = time.time()

    itr = 0

    while diminish_coef or itr < 4:
        ## ADO part1 : STRidge training

        print('itr' + str(itr+1))
        print('Training parameters (STRidge):')

        x = torch.matmul(N_c, P_HY[:, 0]).cpu().detach().numpy()
        y = torch.matmul(N_c, P_HY[:, 1]).cpu().detach().numpy()

        phi = np.zeros([t_c_len, num_term])
        for i in range(num_term):
            # seems not useful to include replace
            phi[:, i] = eval(polynomial_library[i].replace('torch.', ''))
        Y_spline = torch.matmul(N_dt, P_HY).cpu().detach().numpy()

        A_raw[:, 0], tol_best[0] = TrainSTRidge(phi, Y_spline[:,0], lam, eta, d_tol, maxit = 500)
        #pdb.set_trace()
        A_raw[:, 1], tol_best[1] = TrainSTRidge(phi, Y_spline[:,1], lam, eta, d_tol, maxit = 500)   

        print('best tolerance threhold is', tol_best)
        print('prune number of terms to', np.count_nonzero(A_raw))
        #if np.count_nonzero(A_raw) == 4:
            #pdb.set_trace
        #print()


        ## ADO part2 : Spline training
        print('Spline training')

        function_x = function_y = ''
        sparse_c_lst = []
        for i in range(0, num_term):
            term = polynomial_library[i]
            if A_raw[i, 0]!= 0:
                function_x += (' + cx'+str(i)+'*'+term)
                sparse_c_lst.append(globals()['cx'+str(i)])
            if A_raw[i, 1]!= 0:
                function_y += (' + cy'+str(i)+'*'+term)
                sparse_c_lst.append(globals()['cy'+str(i)])
        function_x = function_x[3:]
        function_y = function_y[3:]

        learning_rate = 0.05
        parameters = [{'params': [P_HY]},
                {'params': sparse_c_lst}]

        optimizer = torch.optim.Adam(parameters,lr=learning_rate)

        scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 200, min_lr = 0.0001)

        err_best = 10000
        min_loss = 10000
        epochs_no_improve = 0

        loss_his = []

        for t in range(20000):
            optimizer.zero_grad()
            loss = 0
            #loss = loss_func(output1[:,0],label[:,0])+loss_func(output2[:,0],label[:,1])
            loss1, _,_ = loss_func(N, P_HY, measurement)
            #pdb.set_trace()

            x = torch.matmul(N_c, P_HY[:, 0])
            y = torch.matmul(N_c, P_HY[:, 1])
            rhs_x = eval(function_x)
            rhs_y = eval(function_y)
            #loss2, _,_ = loss_func1(rhs_x,rhs_y, N_dt, P_HY)
            loss2,_,_ = loss_func1(rhs_x, N_dt, P_HY, rhs_y = rhs_y)
            
            loss = loss1+1/T_scale*loss2
            loss.backward()

            scheduler.step(loss)

            optimizer.step()

            if loss.item() >= min_loss:
                epochs_no_improve += 1
            else:
                min_loss = loss.item()
                epochs_no_improve = 0
            
            if epochs_no_improve == no_improve_val and optimizer.param_groups[0]['lr'] == 0.0001:
                print('epoch :', t , 'loss :', loss.item(), 'lrP :', optimizer.param_groups[0]['lr'], ' lrC', optimizer.param_groups[1]['lr'])
                print('Early stopping')
                
                break
            
            if t%5000 == 0:
                print('epoch :', t, 'loss :', loss.item(), 'lrP :', optimizer.param_groups[0]['lr'], 'lrC', optimizer.param_groups[1]['lr'])
            
            
            optimizer.step()

            if t == 19999:
                print('epoch :', t, 'loss :', loss.item(), 'lrP :', optimizer.param_groups[0]['lr'], 'lrC', optimizer.param_groups[1]['lr'])
                print('20000 epochs completed')
        
        for i in range(num_term):
            if A_raw[i, 0]!= 0: A_raw[i, 0] = globals()['cx'+str(i)].cpu().detach().numpy()[0]
            if A_raw[i, 1]!= 0: A_raw[i, 1] = globals()['cy'+str(i)].cpu().detach().numpy()[0]


        for i in range(A_raw.shape[0]):
            for j in range(A_raw.shape[1]):
                if abs(A_raw[i, j]) < tol:
                    A_raw[i, j] = 0
        
        print('prune number of terms to', np.count_nonzero(A_raw))
        ########
        print('in the middle, nonzeros values are \n')
        print()
        function_x = function_y = ''
        for i in range(0, num_term):
            term = polynomial_library[i]
            if A_raw[i,0] != 0: function_x += (' + '+str(np.round(A_raw[i,0], 4))+'*'+term)
            if A_raw[i,1] != 0: function_y += (' + '+str(np.round(A_raw[i,1], 4))+'*'+term)
        print()
        print('x/=', function_x[3:].replace('+ -', '- '))
        print()
        print('y/=', function_y[3:].replace('+ -', '- '))
        print()
        ##########


        ## this one needs to be modified
        loss_HY.append(loss.item() + eta*np.count_nonzero(A_raw))
        terms_HY.append(np.count_nonzero(A_raw))
        #if loss_HY[-1] < loss_HY_min:
        A_raw_HY = A_raw.copy()
        loss_HY_min = loss_HY[-1]

        if np.count_nonzero(A_raw) < num_terms:
            num_terms = np.count_nonzero(A_raw)
        else:
            diminish_coef = False

        itr += 1
        print()
        function_x = function_y = ''

        for i in range(0, num_term):
            term = polynomial_library[i]
            if A_raw_HY[i,0] != 0: function_x += (' + '+str(np.round(A_raw_HY[i,0], 4))+'*'+term)
            if A_raw_HY[i,1] != 0: function_y += (' + '+str(np.round(A_raw_HY[i,1], 4))+'*'+term)
        print()
        print('x/=', function_x[3:].replace('+ -', '- '))
        print()
        print('y/=', function_y[3:].replace('+ -', '- '))
        print()

    print('reach convergence of number of terms in governing equations!')
    print("--- %s seconds ---" % (time.time() - start_time))
    print()
    print('final result :')


    function_x = function_y = ''

    for i in range(0, num_term):
        term = polynomial_library[i]
        if A_raw_HY[i, 0] != 0: function_x += (' + '+str(np.round(A_raw_HY[i,0], 4))+'*'+term)
        if A_raw_HY[i, 1] != 0: function_y += (' + '+str(np.round(A_raw_HY[i,1], 4))+'*'+term)
    print()
    print('x/=', function_x[3:].replace('+ -', '- '))
    print()
    print('y/=', function_y[3:].replace('+ -', '- '))
    print()

    np.savez(caseId+'ADOCoeff',A_raw_HY = A_raw_HY.copy(),P_HY = P_HY.cpu().detach().numpy())

    save_checkpoint(
                dir = savedir,
                epoch = epoch,
                name="ADOswag_checkpoint",
                state_dict=swag_model.state_dict(),
                )
    ## calculate rmse
    rmse = np.linalg.norm(Xi_base-A_raw_HY)/np.linalg.norm(Xi_base)

    numerator = np.count_nonzero((Xi_base*A_raw_HY))
    denom1 = np.count_nonzero(Xi_base)
    denom2 = np.count_nonzero(A_raw_HY)
    ## count the nonzero constant

    
    ##
    Mr = numerator/denom1
    Mp = numerator/denom2
    print('precision is ', Mp)
    print('recall is ', Mr)

    print('rmse is', rmse)
pdb.set_trace()



    



