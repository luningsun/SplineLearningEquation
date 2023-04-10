import torch
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt

import numpy as np

## utils for spline
class SplineDataLikelihood:
    def __init__(self, noise_var):
        print('using data spline loss')
        self.mse = torch.nn.functional.mse_loss
        self.noise_var = noise_var
    def __call__(self, N, P, measurement, h = None,device = 'cuda:0',lnoise = None):
        loss = 0
        MSE = 0
        #pdb.set_trace()
        # actual observed variables
        tmpoutput = torch.matmul(N,P)
        if h is not None:
            tmpoutput = torch.matmul(h,tmpoutput)
        #tmpoutput = h(tmpinpt, device = device)

        output1 = tmpoutput[:,0]
        if lnoise is not None:
            #print('apply noise')
            output1 += lnoise[:,0]
        target1 = measurement[:, 0]
        mse1 = self.mse(output1, target1)#
        MSE += mse1
        loss += mse1/(2*self.noise_var)

        if tmpoutput.shape[1] > 1:
            output2 = tmpoutput[:,1]
            target2 = measurement[:, 1]
            mse2 = self.mse(output2, target2)
            MSE += mse2
            loss += mse2/(2*self.noise_var)
        
        if tmpoutput.shape[1] > 2:
            output3 = tmpoutput[:,2]
            target3 = measurement[:, 2]
            mse3 = self.mse(output3, target3)
            MSE += mse3
            loss += mse3/(2*self.noise_var)
        if tmpoutput.shape[1] > 3:
             output4 = tmpoutput[:,3]
             target4 = measurement[:, 3]
             mse4 = self.mse(output4, target4)
             MSE += mse4
             loss += mse4/(2*self.noise_var)
        
        if tmpoutput.shape[1] > 4:
             output5 = tmpoutput[:,4]
             target5 = measurement[:, 4]
             mse5 = self.mse(output5, target5)
             MSE += mse5
             loss += mse5/(2*self.noise_var)
        
        if tmpoutput.shape[1] > 5:
             output6 = tmpoutput[:,5]
             target6 = measurement[:, 5]
             mse6 = self.mse(output6, target6)
             MSE += mse6
             loss += mse6/(2*self.noise_var) 
        
        return loss, MSE, {"mse", MSE}

class SplineEqLikelihood:
    def __init__(self, noise_var):
        print('using eq spline loss')
        self.mse = torch.nn.functional.mse_loss
        self.noise_var = noise_var

    def __call__(self, rhs_x, N_dt, P, rhs_y = None, rhs_z = None, rhs_x4 = None, rhs_x5 = None, rhs_x6 = None):
        #x = torch.matmul(N_c, P[:, 0])
        #y = torch.matmul(N_c, P[:, 1])
        loss = 0
        MSE = 0
        output1 = torch.matmul(N_dt, P[:, 0])
        target1 = rhs_x
        mse1 = self.mse(output1, target1)
        MSE += mse1
        loss += mse1/(2*self.noise_var)

        if P.shape[1] > 1:
            output2 = torch.matmul(N_dt, P[:, 1])
            #pdb.set_trace()
            target2 = rhs_y
            mse2 = self.mse(output2, target2)
            MSE += mse2
            loss += mse2/(2*self.noise_var)

        if P.shape[1] > 2:
            output3 = torch.matmul(N_dt, P[:, 2])
            target3 = rhs_z
            mse3 = self.mse(output3, target3)
            MSE += mse3
            loss += mse3/(2*self.noise_var)
        
        if P.shape[1] > 3:
             output4 = torch.matmul(N_dt, P[:, 3])
             target4 = rhs_x4
             mse4 = self.mse(output4, target4)
             MSE += mse4
             loss += mse4/(2*self.noise_var)
        
        if P.shape[1] > 4:
             output5 = torch.matmul(N_dt, P[:, 4])
             target5 = rhs_x5
             mse5 = self.mse(output5, target5)
             MSE += mse5
             loss += mse5/(2*self.noise_var)
        
        if P.shape[1] > 5:
             output6 = torch.matmul(N_dt, P[:, 5])
             target6 = rhs_x6
             mse6 = self.mse(output6, target6)
             MSE += mse6
             loss += mse6/(2*self.noise_var)
        
        return loss, MSE, {"mse": MSE}

## utils for NN

class MSEloss:
    def __init__(self):
        print('using det dataloss')
        self.mse = torch.nn.functional.mse_loss
    def __call__(self, model, input, target):
        output = model(input)
        #pdb.set_trace()    
        mse = self.mse(output, target)
        loss = mse
        return loss, output, {"mse": mse}


class EqMSEloss:
    def __init__(self,device = 'cuda:0'):
        print('use det eqloss')
        self.mse = torch.nn.functional.mse_loss
    def _eqres(self,p,output,deriv,porder = 3):
        #print('porder is', porder)
        #pdb.set_trace()
        eqres1 = deriv[:,0]-(p[0,0]*output[:,0]**porder+p[0,1]*output[:,1]**porder)
        eqres2 = deriv[:,1]-(p[1,0]*output[:,0]**porder+p[1,1]*output[:,1]**porder)
        #pdb.set_trace()
        return torch.column_stack((eqres1,eqres2))

    def __call__(self, p, output, deriv, porder = 3):
        eqres = self._eqres(p, output, deriv, porder = porder)
        loss = 0
        MSE = 0
        #pdb.set_trace()
        for i in range(eqres.shape[1]):
            mse = self.mse(eqres[:,i], torch.zeros_like(eqres[:,i]))
            MSE += mse
            loss += mse
        return loss, eqres, {"mse": MSE}


class GaussianLikelihood:
    """
    Minus Gaussian likelihood for regression problems.

    Mean squared error (MSE) divided by `2 * noise_var`.
    """
    
    def __init__(self, noise_var = 0.5,est_noise = False, device = 'cuda:0'):
        self.noise_var = torch.from_numpy(noise_var).to(device)
        if est_noise == True:
            assert self.noise_var.requires_grad == True
            print('est_noise is', est_noise)
            print('noise_var requires grad is', self.noise_var.requires_grad)
        self.mse = torch.nn.functional.mse_loss
    
    def __call__(self, model, input, target):

        output = model(input)
        
        if self.noise_var is not None:
            mse = self.mse(output[:,0], target[:,0],reduction = 'none')
            #pdb.set_trace()
            loss = mse / (2 * self.noise_var)

            #mse2 = self.mse(output[:,1], target[:,1],reduction = 'none')
            return loss.mean(), output, {"mse": mse}
        
        else:
            #pdb.set_trace()
            mean = output[:,0].view_as(target)
            var = output[:,1].view_as(target)

            mse = self.mse(mean, target, reduction='none')
            mean_portion = mse / (2 * var)
            ## doesn't need this?
            var_portion = 0.5 * torch.log(var)
            loss = mean_portion + var_portion

            return loss.mean(), output[:,0], {'mse': torch.mean((mean - target)**2.0)}


class EqGaussianLikelihood:
    """
    Minus Gaussian likelihood for regression problems.

    Mean squared error (MSE) divided by `2 * noise_var`.
    """
    def __init__(self, noise_var = 0.5,device = 'cuda:0'):
        
        self.noise_var = torch.from_numpy(np.array(noise_var)).to(device)
        #pdb.set_trace()
        self.mse = torch.nn.functional.mse_loss
    def _eqres(self,p,output,deriv,porder = 3):
        eqres1 = deriv[:,0]-(p[0][0]*output[:,0]**porder+p[0][1]*output[:,1]**porder)
        eqres2 = deriv[:,1]-(p[1][0]*output[:,0]**porder+p[1][1]*output[:,1]**porder)

        #return [eqres1,eqres2]
        return torch.column_stack((eqres1, eqres2))

    def __call__(self, p, output, deriv, porder = 3):
        #print('weight is ', weight)
        
        #output = torch.matmul(libs,weight)
        eqres = self._eqres(p, output, deriv, porder = porder)
        
        
        loss = 0
        MSE = 0
        for i in range(eqres.shape[1]):
            mse = self.mse(eqres[:,i], torch.zeros_like(eqres[:,i]))
            MSE += mse
            #pdb.set_trace()
            loss += mse / (2 * self.noise_var)
            #pdb.set_trace()
            #pdb.set_trace()
            #loss += 0.5*torch.log(self.noise_var)
        #pdb.set_trace()
        return loss, eqres, {"mse": MSE}
        


class dataPriors:
    def __init__(self, w_prior_rate, w_prior_shape, estNoise = False,device = 'cpu', log_beta = None, beta_prior_shape = None, beta_prior_rate = None):
        print('init data priors')
        self.w_prior_rate = w_prior_rate
        self.w_prior_shape = w_prior_shape

        self.log_beta = log_beta
        assert self.log_beta.requires_grad == True
        self.beta_prior_rate = beta_prior_rate
        self.beta_prior_shape = beta_prior_shape
        
        self.device = device
        self.estNoise = estNoise
        self.ctr = 0
    def __call__(self,model):
        if (self.ctr+1)% 1000 == 0:
            print('log_beta is', self.log_beta)
        self.ctr += 1
        log_prob_prior_w = torch.tensor(0.).to(self.device)
        # log prob of prior of weights, i.e. log prob of studentT

        for param in model.parameters():
            log_prob_prior_w += \
                    torch.log1p(0.5 /self.w_prior_rate * param.pow(2)).sum()
            #pdb.set_trace()
        log_prob_prior_w *= -(self.w_prior_shape + 0.5)
        num_parameters = model._num_parameters()
        #pdb.set_trace()
        ## convert it to mse, divide by num of parameters
        log_prob_prior_w /= num_parameters
        # Log Gamma Output-wise noise prior
        if self.estNoise == True:
            prior_log_beta = ((self.beta_prior_shape - 1.0) * self.log_beta \
                            - self.log_beta.exp() * self.beta_prior_rate)
            return -log_prob_prior_w - prior_log_beta
        else:
            return -log_prob_prior_w
class eqPriors:
    def __init__(self,w_prior_rate, w_prior_shape, estNoise = False,device = 'cpu', log_beta = None, beta_prior_shape = None, beta_prior_rate = None):
        print('init eq priors')
        self.w_prior_rate = w_prior_rate
        self.w_prior_shape = w_prior_shape

        self.log_beta = log_beta
        self.beta_prior_rate = beta_prior_rate
        self.beta_prior_shape = beta_prior_shape
        
        self.device = device
        self.estNoise = estNoise
    def __call__(self,weights):
        log_prob_prior_w = torch.tensor(0.).to(self.device)
        # log prob of prior of weights, i.e. log prob of studentT
        for param in weights:
            
            log_prob_prior_w += \
                    torch.log1p(0.5 /self.w_prior_rate * param.pow(2)).sum()
            #pdb.set_trace()
        log_prob_prior_w *= -(self.w_prior_shape + 0.5)

        num_parameters = weights.numel()
        #pdb.set_trace()
        ## convert it to mse, divide by num of parameters
        log_prob_prior_w /= num_parameters
        # Log Gamma Output-wise noise prior
        if self.estNoise == True:
            prior_log_beta = ((self.beta_prior_shape - 1.0) * self.log_beta \
                            - self.log_beta.exp() * self.beta_prior_rate)
            return -log_prob_prior_w - prior_log_beta
        else:
            return -log_prob_prior_w


def cross_entropy(model, input, target):
    # standard cross-entropy loss function

    output = model(input)

    loss = F.cross_entropy(output, target)

    return loss, output, {}


def cross_entropy_output(output, target):
    # standard cross-entropy loss function

    loss = F.cross_entropy(output, target)

    return loss, {}


def adversarial_cross_entropy(model, input, target, lossfn = F.cross_entropy, epsilon = 0.01):
    # loss function based on algorithm 1 of "simple and scalable uncertainty estimation using
    # deep ensembles," lakshminaraynan, pritzel, and blundell, nips 2017, 
    # https://arxiv.org/pdf/1612.01474.pdf
    # note: the small difference bw this paper is that here the loss is only backpropped
    # through the adversarial loss rather than both due to memory constraints on preresnets
    # we can change back if we want to restrict ourselves to VGG-like networks (where it's fine).

    #scale epsilon by min and max (should be [0,1] for all experiments)
    #see algorithm 1 of paper
    scaled_epsilon = epsilon * (input.max() - input.min())

    #force inputs to require gradient
    input.requires_grad = True

    #standard forwards pass
    output = model(input)
    loss = lossfn(output, target)

    #now compute gradients wrt input
    loss.backward(retain_graph = True)
        
    #now compute sign of gradients
    inputs_grad = torch.sign(input.grad)
    
    #perturb inputs and use clamped output
    inputs_perturbed = torch.clamp(input + scaled_epsilon * inputs_grad, 0.0, 1.0).detach()
    #inputs_perturbed.requires_grad = False

    input.grad.zero_()
    #model.zero_grad()

    outputs_perturbed = model(inputs_perturbed)
    
    #compute adversarial version of loss
    adv_loss = lossfn(outputs_perturbed, target)

    #return mean of loss for reasonable scalings
    return (loss + adv_loss)/2.0, output, {}

def masked_loss(y_pred, y_true, void_class = 11., weight=None, reduce = True):
    # masked version of crossentropy loss

    el = torch.ones_like(y_true) * void_class
    mask = torch.ne(y_true, el).long()

    y_true_tmp = y_true * mask

    loss = F.cross_entropy(y_pred, y_true_tmp, weight=weight, reduction='none')
    loss = mask.float() * loss

    if reduce:
        return loss.sum()/mask.sum()
    else:
        return loss, mask

def seg_cross_entropy(model, input, target, weight = None):
    output = model(input)

    # use masked loss function
    loss = masked_loss(output, target, weight=weight)

    return {'loss': loss, 'output': output}

def seg_ale_cross_entropy(model, input, target, num_samples = 50, weight = None):
        #requires two outputs for model(input)

        output = model(input)
        mean = output[:, 0, :, :, :]
        scale = output[:, 1, :, :, :].abs()

        output_distribution = torch.distributions.Normal(mean, scale)

        total_loss = 0

        for _ in range(num_samples):
                sample = output_distribution.rsample()

                current_loss, mask = masked_loss(sample, target, weight=weight, reduce=False)
                total_loss = total_loss + current_loss.exp()
        mean_loss = total_loss / num_samples

        return {'loss': mean_loss.log().sum() / mask.sum(), 'output': mean, 'scale': scale}
