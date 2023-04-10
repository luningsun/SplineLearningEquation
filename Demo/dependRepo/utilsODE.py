import numpy as np
import torch
import pdb
import os


#%% Define functions here
# =============================================================================
# Define the ODE of Van der Pol
# =============================================================================
def  VanderPoluq(u,t,p):
    du1=p[1]*u[1]
    du2=p[2]*u[1]+p[3]*u[0]**2*u[1]+p[0]*u[0]
    
    du=np.array([du1,du2])
    #pdb.set_trace()
    #p0=0.5
    
    return du

#%% Define functions here
# =============================================================================
# Define the ODE of Van der Pol
# =============================================================================
def euler_step(u, f, dt, *args):
    """Returns the solution at the next time-step using 
    the forward Euler method.
    
    Parameters
    ----------
    u : array of floats
        solution at the current time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    args : optional arguments for the rhs function
    
    Returns
    -------
    unp1 : array of floats
        approximate solution at the next time step.
    """
    unp1 = u + dt * f(u, *args)
    return unp1



def VanderPol(u,t,p):
    du1=u[1]
    du2=p[0]*(1-u[0]**2)*u[1]-u[0]
    
    du=np.array([du1,du2])
    
    #p0=0.5
    
    return du



def VanderPolUQ(u,t, A, poly_lib):
    N = A.shape[1]
    d = np.zeros(np.shape(u))
    x = u[0]
    y = u[1]

    for i in range(len(poly_lib)):
        for j in range(len(u)):
            if A[i,j]!=0:
                d[j]+=A[i,j]*eval(poly_lib[i])
    
    #p0=0.5
    
    return d
# =============================================================================
# Define the ODE of Lorenz96 (Modified from https://en.wikipedia.org/wiki/Lorenz_96_model)
# =============================================================================
def Lorenz96(x, t, p):
    # Lorenz 96 model
    # Compute state derivatives
    #pdb.set_trace()
    N=p[0]
    d = np.zeros(np.shape(x))
    # First the 3 edge cases: i=1,2,N
    d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
    d[1] = (x[2] - x[N-1]) * x[0] - x[1]
    d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
    # Then the general case
    for i in range(2, N-1):
        d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    # Add the forcing term
    d = d + p[1]
    
    #p0=[10,8]
    
    # Return the state derivatives
    return d
def Lorenz96UQ(x,t,A,poly_lib):
    N = A.shape[1]
    d = np.zeros(np.shape(x))
    for j in range(len(x)):
        locals()['x'+str(j+1)] = x[j] 

    for i in range(len(poly_lib)):
        for j in range(len(x)):
            if A[i,j]!=0:
                d[j]+=A[i,j]*eval(poly_lib[i])

    return d

# =============================================================================
# Define the ODE for the Lorenz system
# =============================================================================
def Lorenz(u, t, p):
   du1 = p[0]*u[0]+p[1]*u[1]
   du2 = p[2]*u[0]+p[3]*u[0]*u[2]+p[4]*u[1]
   du3 = p[5]*u[0]*u[1]+p[6]*u[2]
   
   du=np.array([du1,du2,du3])
   
   #p0=[-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3]
   
   return du



def CubicOsc(u,t,p):
    du1=p[0]*u[0]**3+p[1]*u[1]**3
    du2=p[2]*u[0]**3+p[3]*u[1]**3
    
    du=np.array([du1,du2])
    
    #p0=[-0.1,2,-2,-0.1]
    
    return du 


def LinearOsc(u,t,p):
    du1=p[0]*u[0]+p[1]*u[1]
    du2=p[2]*u[0]+p[3]*u[1]
    
    du=np.array([du1,du2])
    
    #p0=[-0.1,2,-2,-0.1]
    return du

def ConstOsc(u,t,p):
    du1 = p[0]*1+p[1]*1
    du2 = p[2]*1+p[3]*1

    du=np.array([du1,du2])

    return du

def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

def set_weights(model, vector, device=None):
    offset = 0
    for param in model:
        #pdb.set_trace()
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()


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


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



