import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm
from collections import defaultdict
from time import gmtime, strftime
import sys
import pdb
import torch.nn.functional as F
import matplotlib.pyplot as plt

class normalizer(object):
	def __init__(self, mean,std,normalizeFlag = True):
		self.mean = mean
		self.std  = std
		self.normalizeFlag = normalizeFlag
	def N(self, u):
		if self.normalizeFlag:
			return (u - self.mean)/self.std
		else:
			return u
	def Ninv(self, u):
		if self.normalizeFlag:
			return u*self.std + self.mean
		else:
			return u



def get_logging_print(fname):
    cur_time = strftime("%m-%d_%H:%M:%S", gmtime())

    def print_func(*args):
        str_to_write = ' '.join(map(str, args))
        filename = fname % cur_time if '%s' in fname else fname
        with open(filename, 'a') as f:
            f.write(str_to_write + '\n')
            f.flush()

        print(str_to_write)
        sys.stdout.flush()

    return print_func, fname % cur_time if '%s' in fname else fname


def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i=0
    for tensor in likeTensorList:
        #n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:,i:i+n].view(tensor.shape))
        i+=n
    return outList


def LogSumExp(x,dim=0):
    m,_ = torch.max(x,dim=dim,keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim,keepdim=True))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch=None, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    if epoch is not None:
       name = '%s-%d.pt' % (name, epoch)
    else:
       name = '%s.pt' % (name)
    state.update(kwargs)
    filepath = os.path.join(dir, name)
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer, cuda=True, regression=False, verbose=False, subset=None, regularizer = None, eqregularizer = None, weight = None, libs_flag= 0, eqCirterion = None, inputNormalizer = None, labelNormalizer = None, T_scale = 1, dataFlag = True,epoch = None,est_Noise = False):
    loss_sum = 0.0
    loss_eq_sum = 0.0
    loss_data_sum = 0.0
    loss_eq_prior_sum = 0.0
    loss_data_prior_sum = 0.0

    stats_sum = defaultdict(float)
    correct = 0.0
    verb_stage = 0
    
    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        optimizer.zero_grad()
        #pdb.set_trace()
        loss = 0.0
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        if inputNormalizer is not None and labelNormalizer is not None:
            input_n = inputNormalizer.N(input)
            target_n = labelNormalizer.N(target)
        else:
            input_n = input
            target_n = target
        
        # data loss is the normalized data and labels
        if dataFlag is not False:
            loss, output, stats = criterion(model, input_n, target_n)
            #print('loss is', loss)
            #pdb.set_trace()
            loss_data = loss.clone()


        else:
            loss_data = 0.0
        #pdb.set_trace()
        if libs_flag == 1:
            #print('I am here')
            #libs = (input[:,0]**3).view(-1,1)
            
            if labelNormalizer is not None:

                libs = labelNormalizer.Ninv(model(input_n))
            else:
                libs = model(input_n)
            libs = torch.hstack((libs,libs**2))
            #pdb.set_trace()
        if weight is not None and libs is not None and eqCirterion is not None:
            #pdb.set_trace()
            # equation loss is not normalzied libs and labels
            loss_eq, _, _ = eqCirterion(weight, libs, target)
            
            loss += 1/T_scale*loss_eq
        
        ## added
        if regularizer:
            #print('use data regularizer')
            loss_data_prior = regularizer(model)
            loss += loss_data_prior###
        if eqregularizer:
            #print('use eq regularizer')
            loss_eq_prior = eqregularizer(weight)
            loss += loss_eq_prior
        #pdb.set_trace()
        ## added

        #print('loss_before backprop is', loss)
        #pdb.set_trace()
        # why need to add retain_graph == True, is that a bug?
        #loss.backward(retain_graph = True)
        loss.backward()
        #pdb.set_trace()
        optimizer.step()
        if dataFlag is not False:
            loss_sum += loss.data.item() * input.size(0)
            loss_data_sum += loss_data.data.item()*input.size(0)
        else:
            loss_sum += loss*input.size(0)
            loss_data_sum += loss_data*input.size(0)

        if eqCirterion is not None:
            loss_eq_sum += loss_eq.data.item()*input.size(0)
            loss_eq_prior_sum += loss_eq_prior.data.item()*input.size(0)
        if dataFlag is not False:
            for key, value in stats.items():
                stats_sum[key] += value * input.size(0)
        if est_Noise == True:
            loss_data_prior_sum += loss_data_prior.data.item()*input.size(0)
        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print('Stage %d/10. Loss: %12.4f. Acc: %6.2f' % (
                verb_stage + 1, loss_sum / num_objects_current,
                correct / num_objects_current * 100.0
            ))
            verb_stage += 1
   
    return {
            'loss': loss_sum / num_objects_current,
            'loss_eq': None if eqCirterion is None else loss_eq_sum/num_objects_current,
            'loss_data': None if dataFlag is False else loss_data_sum/num_objects_current,
            'loss_eq_prior': None if eqCirterion is None else loss_eq_prior_sum/num_objects_current,
            'loss_data_prior': None if est_Noise == False else loss_data_prior_sum/num_objects_current,
            'accuracy': None if regression else correct / num_objects_current * 100.0,
            'stats': None if dataFlag is False else {key: value / num_objects_current for key, value in stats_sum.items()}
        }


#def eval(loader, model, criterion, cuda=True, regression=False, verbose=False):
def eval(input, target, model, criterion, cuda=True, regression=False, verbose=False, inputNormalizer = None, labelNormalizer = None,epoch = None,plotdir=None,dataFlag = True,weight = None, libs_flag = 1, eqCirterion = None):
    if dataFlag is False:
        print("warning, training not use data loss, make sure that's what you want")
    
    loss_sum = 0.0
    correct = 0.0
    stats_sum = defaultdict(float)
    num_objects_total = input.shape[0]
    loss = 0.0
    model.eval()

    


    with torch.no_grad():
        #if verbose:
            #loader = tqdm.tqdm(loader)

        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        if inputNormalizer is not None and labelNormalizer is not None:
            input_n = inputNormalizer.N(input)
            target_n = labelNormalizer.N(target)
        else:
            input_n = input
            target_n = target

        if libs_flag == 1:
            #print('I am here')
            #libs = (input[:,0]**3).view(-1,1)
            #libs = input[:,0].view(-1,1)
            if labelNormalizer is not None:
                libs = labelNormalizer.Ninv(model(input_n))
            else:
                libs = model(input_n)
            libs = torch.hstack((libs, libs**2))
        if dataFlag is not False:
            # this output is normalized
            loss, output, stats = criterion(model, input_n, target_n)
        if weight is not None and libs is not None and eqCirterion is not None:
            # this output is not normalized
            loss_eq, outputeq, statseq = eqCirterion(weight, libs, target)
        
        #pdb.set_trace()
        plot_x = input[:,0].view(-1,1).data.numpy()
        if labelNormalizer is not None:
            plot_y1 = labelNormalizer.Ninv(output).data.numpy()
        else:
            plot_y1 = output.data.numpy()
        plot_y2 = target.data.numpy()
        plt.figure()
        
        plt.plot(plot_x,plot_y2,label = 'label')
        plt.plot(plot_x,plot_y1,'x-.',label = 'prediction')
        plt.legend()
        plt.savefig(plotdir+'Epoch'+str(epoch)+'evaluate.png',bbox_inches = 'tight')
        plt.close('all')

        if dataFlag is not False:
            loss_sum += loss.item() * input.size(0)
        else:
            loss_sum += loss*input.size(0)
        if dataFlag is not False:
            for key, value in stats.items():
                stats_sum[key] += value

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / num_objects_total,
        'accuracy': None if regression else correct / num_objects_total * 100.0,
        'stats': None if dataFlag is False else {key: value / num_objects_total for key, value in stats_sum.items()}
    }


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            batch_size = input.size(0)
            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += batch_size

    return {
        'predictions': np.vstack(predictions),
        'targets': np.concatenate(targets)
    }


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)

        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps = 1e-10):
    return torch.log(x/(1.0 - x + eps))


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    #will assume that model is already in eval mode
    #model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor


def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()

def extract_parameters(model):
    params = []	
    for module in model.modules():	
        for name in list(module._parameters.keys()):	
            if module._parameters[name] is None:	
                continue	
            param = module._parameters[name]	
            params.append((module, name, param.size()))	
            module._parameters.pop(name)	
    return params

def set_weights_old(params, w, device):	
    offset = 0
    for module, name, shape in params:
        size = np.prod(shape)	       
        value = w[offset:offset + size]
        setattr(module, name, value.view(shape).to(device))	
        offset += size
