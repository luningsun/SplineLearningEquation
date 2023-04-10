#python pkgs
import sys
import os
import seaborn as sns
#3rd part pkgs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pdb

sys.path.append('./dependRepo/subspace_inference/posteriors/')
from subspaces import Subspace

from utilsODE import *
def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

class MLP(nn.Module):
    def __init__(self,in_n, hid_n, out_n,init_cond = None,end_cond = None):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.features = nn.Sequential()
        self.features.add_module('hidden', nn.Linear(in_n,hid_n))
        self.features.add_module('active1', self.tanh)
        self.features.add_module('hidden2', nn.Linear(hid_n,hid_n))
        self.features.add_module('active2', self.tanh)
        self.features.add_module('hidden3', nn.Linear(hid_n,out_n))
        if init_cond is not None:
            self.init_cond = init_cond
        else:
            self.init_cond = None
        if end_cond is not None:
            self.end_cond = end_cond
        else:
            self.end_cond = None
         
        
    def forward(self,x):
        
        out = self.features(x)
        if self.init_cond is not None and self.end_cond is None:
            #print('imposing inital cond')
            out = out*x+self.init_cond
        elif self.init_cond is not None and self.end_cond is not None:
            # x = 0, out = init_cond
            # x = 25, out = end_cond
            #pdb.set_trace()
            out = -self.end_cond/25*(0-x) + self.init_cond*(25-x)/25 + (0-x)*(25-x)*out
            
        #a_train = self.tanh(x)
        #a_train = 2*self.sigmoid(x)
        return out


class SplineSWAG(torch.nn.Module):

    def __init__(self, paramList, subspace_type,
                 subspace_kwargs=None, var_clamp=1e-6, device = 'cuda:0', *args, **kwargs):
        super(SplineSWAG, self).__init__()
        self.w_collector = []
        self.mean_collector = []
        
        self.num_parameters = sum(param.numel() for param in paramList)

        self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))
        
        ### add by lsun
        for i in range(len(paramList)):
            self.register_buffer('param'+str(i+1), paramList[i])
        self.paramList = paramList
        #pdb.set_trace()
        ###
        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        self.subspace = Subspace.create(subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.var_clamp = var_clamp

        self.cov_factor = None
        self.model_device = device
        

    def collect_model(self, paramList, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        w = flatten([param.detach().cpu() for param in paramList])
        # first moment
        self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.mean.add_(w / (self.n_models.item() + 1.0))

        #self.mean_collector.append(self.mean[-4:,].cpu().detach().numpy().copy())
        #self.w_collector.append(w[-4:,].cpu().detach().numpy().copy())
        # second moment
        self.sq_mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        self.sq_mean.add_(w ** 2 / (self.n_models.item() + 1.0))

        dev_vector = w - self.mean

        self.subspace.collect_vector(dev_vector, *args, **kwargs)
        self.n_models.add_(1)

    def _get_mean_and_variance(self):
        variance = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        self.cov_factor = self.subspace.get_space()

    def sample(self, scale=0.5, diag_noise=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()
        #pdb.set_trace()
        eps_low_rank = torch.randn(self.cov_factor.size()[0])
        z = self.cov_factor.t() @ eps_low_rank
        if diag_noise:
            z += variance * torch.randn_like(variance)
        z *= scale ** 0.5
        sample = mean + z

        # apply to parameters
        set_weights(self.paramList, sample, self.model_device)
        return sample

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()
