from os import XATTR_SIZE_MAX
import sys
import numpy as np
from scipy.integrate import odeint
sys.path.append('/home/luningsun/Documents/Research/ai-equationdiscovery/reference/utilsODE')
from utilsODE import *
import pdb
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
class datasetExamples:
    def __init__(self, noise = None):

        print('init dataset')
        if noise is not None:
            self.noise = noise
    def __call__(self,caseName,A = None, poly_lib = None):
        if caseName == 'toy':
            def features(x):
                return np.hstack([x / 2.0, (x / 2.0) ** 2])

            def func(x):
                return x**3
            def add_noise(y,std = 10):
                noise = std*np.random.randn(*y.shape)
                y+=noise
                return y
            x = np.linspace(-4,4,20).reshape(-1,1)
            f = features(x)
            y = func(x)
            y = add_noise(y,3)

            return x,f,y,features
        elif caseName == 'vanderpol':
            NoisePercentage=self.noise
            def features(x):
                #return np.hstack([x / 2.0, (x / 2.0) ** 2])
                return x
            # Define the parameters
            p0=np.array([0.5])

            # Define the initial conditions
            x0=np.array([-2,1])

            # Define the time points
            T=10.0
            dt=0.01

            t=np.linspace(0.0,T,int(T/dt))

            # Now simulate the system
            x=odeint(VanderPol,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
            dx=np.transpose(VanderPol(np.transpose(x), 0, p0))

            # Get the size info
            stateVar,dataLen=np.transpose(x).shape

            # Generate the noise
            NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
            Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])

            # Add the noise and get the noisy data
            xn=x+Noise
            
            # Test the SINDy
            N_SINDy_Iter=15
            disp=0
            NormalizeLib=0
            libOrder=3
            lam=0.1
            t = t.reshape(-1,1)
            f = t
            return t, f, xn, features
        elif caseName == 'lorenz96':
            NoisePercentage=self.noise
            def features(x):
                #return np.hstack([x / 2.0, (x / 2.0) ** 2])
                return x
            np.random.seed(4)
            p0=np.array([6,8])
            x0=p0[1]*np.ones(p0[0])
            x0[0]=1

            T=20.0
            dt=0.01
            t=np.linspace(0.0,T,int(T/dt))
            
            # test evaluate self written system
            if A is not None and poly_lib is not None:
                xtt = odeint(Lorenz96UQ, x0, t, args = (A, poly_lib),rtol = 1e-12, atol = 1e-12)
            
            # Now simulate the system
            x=odeint(Lorenz96,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)

            if A is not None and poly_lib is not None:
                rmse = np.linalg.norm(x-xtt)
                for k in range(len(x0)):
                    plt.figure()
                    plt.plot(x[:,k],'r')
                    plt.plot(xtt[:,k],'b-.')
                    plt.ylabel('x'+str(k+1))
                    plt.savefig('testx'+str(k+1),bbox_inches = 'tight')
                plt.close('all')
            #pdb.set_trace()
            
            dx=np.transpose(Lorenz96(np.transpose(x), 0, p0))
            # Get the size info
            stateVar,dataLen=np.transpose(x).shape
            # Generate the noise
            
            NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
            Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])
            # Add the noise and get the noisy data
            xn=x+Noise
            # Test the SINDy
            N_SINDy_Iter=15
            disp=0
            NormalizeLib=0
            libOrder=3
            lam=0.1
            t = t.reshape(-1,1)
            f = t
            #pdb.set_trace()
            
            return t, f, xn, features
        elif caseName == 'lorenz63':
            #%% Define how many percent of noise you need
            NoisePercentage=self.noise
            def features(x):
                #return np.hstack([x / 2.0, (x / 2.0) ** 2])
                return x
            #%% Simulate
            # Define the random seed for the noise generation
            np.random.seed(4)

            # Define the parameters
            p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])

            # Define the initial conditions
            x0=np.array([5.0,5.0,25.0])

            # Define the time points
            T=20.0
            dt=0.01

            t=np.linspace(0.0,T,int(T/dt))

            # Now simulate the system
            x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
            dx=np.transpose(Lorenz(np.transpose(x), 0, p0))

            # Get the size info
            stateVar,dataLen=np.transpose(x).shape

            # Generate the noise
            if NoisePercentage is not None:
                NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
                Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])
            else:
                Noise = 0
                print('noise is ', Noise)
            # Add the noise and get the noisy data
            xn=x+Noise
            
            # Test the SINDy
            N_SINDy_Iter=15
            disp=0
            NormalizeLib=0
            libOrder=2
            lam=0.2

            t = t.reshape(-1,1)
            f = t
            return t, f, xn, features
        elif caseName == 'cubic':
            np.random.seed(5)
            
            NoisePercentage = self.noise
            #NoisePercentage=0.01
            def features(x):
                #return np.hstack([x / 2.0, (x / 2.0) ** 2])
                return x
            # Define the parameters
            p0=np.array([-0.1,2,-2,-0.1])#/100

            # Define the initial conditions
            x0=np.array([0,2])

            # Define the time points
            #T=25.0
            T = 20.0
            dt=0.01

            t=np.linspace(0.0,T,int(T/dt)+1)

            # Now simulate the system
            x=odeint(CubicOsc,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
            dx=np.transpose(CubicOsc(np.transpose(x), 0, p0))

            # Get the size info
            stateVar,dataLen=np.transpose(x).shape

            # Generate the noise
            if NoisePercentage is not None:
                NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
                Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])
            else:
                
                Noise = 0
                print('noise is ', Noise)
            # Add the noise and get the noisy data
            xn=x+Noise

            # Test the SINDy
            N_SINDy_Iter=15
            disp=0
            NormalizeLib=0
            libOrder=3
            lam=0.08
            t = t.reshape(-1,1)
            #f = features(t)
            f = t
            return t, f, xn, features
        elif caseName == 'linear':
            np.random.seed(5)
            #NoisePercentage=30
            NoisePercentage=0.01
            def features(x):
                #return np.hstack([x / 2.0, (x / 2.0) ** 2])
                return x
            # Define the parameters
            p0=np.array([-0.1,2,-2,-0.1])

            # Define the initial conditions
            x0=np.array([0,2])

            # Define the time points
            T=25.0
            dt=0.01

            t=np.linspace(0.0,T,int(T/dt))

            # Now simulate the system
            x=odeint(LinearOsc,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
            dx=np.transpose(LinearOsc(np.transpose(x), 0, p0))

            # Get the size info
            stateVar,dataLen=np.transpose(x).shape

            # Generate the noise
            NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
            Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])

            # Add the noise and get the noisy data
            xn=x+Noise

            # Test the SINDy
            N_SINDy_Iter=15
            disp=0
            NormalizeLib=0
            libOrder=3
            lam=0.08
            t = t.reshape(-1,1)
            #f = features(t)
            f = t
            return t, f, xn, features
        elif caseName == 'constant':
            np.random.seed(5)
            #NoisePercentage=30
            NoisePercentage=0.01
            def features(x):
                #return np.hstack([x / 2.0, (x / 2.0) ** 2])
                return x
            # Define the parameters
            p0=np.array([-0.1,2,-2,-0.1])/100

            # Define the initial conditions
            x0=np.array([0,2])

            # Define the time points
            T=25.0
            dt=0.01

            t=np.linspace(0.0,T,int(T/dt))

            # Now simulate the system
            x=odeint(ConstOsc,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
            dx=np.transpose(ConstOsc(np.transpose(x), 0, p0))

            # Get the size info
            stateVar,dataLen=np.transpose(x).shape

            # Generate the noise
            NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
            Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])

            # Add the noise and get the noisy data
            xn=x+Noise

            # Test the SINDy
            N_SINDy_Iter=15
            disp=0
            NormalizeLib=0
            libOrder=3
            lam=0.08
            t = t.reshape(-1,1)
            #f = features(t)
            f = t
            return t, f, xn, features

        else:
            raise NotImplementedError('this case is not supported')


def UQCubic(p0):
    np.random.seed(5)
            
    def features(x):
        #return np.hstack([x / 2.0, (x / 2.0) ** 2])
        return x
    # Define the parameters
    #p0=np.array([-0.1,2,-2,-0.1])#/100
    #p0 = np.transpose(p0)
    # Define the initial conditions
    x0=np.array([0,2])

    # Define the time points
    #T=25.0
    T = 20.0
    dt=0.01

    t=np.linspace(0.0,T,int(T/dt)+1)

    # Now simulate the system
    x=odeint(CubicOsc,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
    dx=np.transpose(CubicOsc(np.transpose(x), 0, p0))

    # Get the size info
    stateVar,dataLen=np.transpose(x).shape

    # Add the noise and get the noisy data
    xn=x

    # Test the SINDy
    N_SINDy_Iter=15
    disp=0
    NormalizeLib=0
    libOrder=3
    lam=0.08
    t = t.reshape(-1,1)
    #f = features(t)
    f = t
    return t, f, xn, features

def UQLorenz63(p0):
    #%% Define how many percent of noise you need
    def features(x):
        #return np.hstack([x / 2.0, (x / 2.0) ** 2])
        return x
    #%% Simulate
    # Define the random seed for the noise generation
    np.random.seed(4)

    # Define the parameters
    #p0=np.array([-10.0,10.0,28.0,-1.0,-1.0,1.0,-8/3])
    pdb.set_trace()
    # Define the initial conditions
    x0=np.array([5.0,5.0,25.0])

    # Define the time points
    T=20.0
    dt=0.01

    t=np.linspace(0.0,T,int(T/dt))

    # Now simulate the system
    x=odeint(Lorenz,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
    dx=np.transpose(Lorenz(np.transpose(x), 0, p0))

    # Get the size info
    stateVar,dataLen=np.transpose(x).shape

    # Add the noise and get the noisy data
    xn=x
    
    # Test the SINDy
    N_SINDy_Iter=15
    disp=0
    NormalizeLib=0
    libOrder=2
    lam=0.2

    t = t.reshape(-1,1)
    f = t
    return t, f, xn, features
def UQLorenz96(A = None, poly_lib = None):
    def features(x):
        #return np.hstack([x / 2.0, (x / 2.0) ** 2])
        return x
    np.random.seed(4)
    p0=np.array([6,8])
    x0=p0[1]*np.ones(p0[0])
    x0[0]=1

    T=20.0
    dt=0.01
    t=np.linspace(0.0,T,int(T/dt))
    
    # test evaluate self written system
    if A is not None and poly_lib is not None:
        xtt = odeint(Lorenz96UQ, x0, t, args = (A, poly_lib),rtol = 1e-12, atol = 1e-12)
    
    # Now simulate the system
    #dx=np.transpose(Lorenz96(np.transpose(x), 0, p0))
    # Get the size info
    stateVar,dataLen=np.transpose(xtt).shape

    # Add the noise and get the noisy data
    xn=xtt
    # Test the SINDy
    N_SINDy_Iter=15
    disp=0
    NormalizeLib=0
    libOrder=3
    lam=0.1
    t = t.reshape(-1,1)
    f = t
    return t, f, xn, features


def UQVanderPol(A = None, poly_lib = None):
    np.random.seed(5)
            
    def features(x):
        #return np.hstack([x / 2.0, (x / 2.0) ** 2])
        return x
    # Define the parameters
    #p0=np.array([-0.1,2,-2,-0.1])#/100
    # Define the initial conditions
    x0=np.array([-2,1])

    # Define the time points
    T=10.0
    dt=0.01

    t=np.linspace(0.0,T,int(T/dt))

    if A is not None and poly_lib is not None:
        xtt = odeint(VanderPolUQ, x0, t, args = (A, poly_lib),rtol = 1e-12, atol = 1e-12)
    #dx=np.transpose(VanderPoluq(np.transpose(x), 0, p0))

    # Get the size info
    stateVar,dataLen=np.transpose(xtt).shape

    # Add the noise and get the noisy data
    xn=xtt
    
    # Test the SINDy
    N_SINDy_Iter=15
    disp=0
    NormalizeLib=0
    libOrder=3
    lam=0.1
    t = t.reshape(-1,1)
    f = t
    return t, f, xn, features
    
def UQAdvection(c = 1):
    np.random.seed(5)

    def rhs_backward_periodic(u, dx, c):
        """Returns the right-hand side of the wave
        equation based on backward finite differences
        
        Parameters
        ----------
        u : array of floats
            solution at the current time-step.
        dx : float
            grid spacing
        c : float
            advection velocity
        
        Returns
        -------
        f : array of floats
            right-hand side of the wave equation with
            boundary conditions implemented
        """
        nx = u.shape[0]
        f = np.empty(nx)
        f[1:] = -c*(u[1:]-u[0:-1]) / dx
        f[0] = f[-1] # alternatively: f[0] = -c*(u[0]-u[-2]) / dx
        
        return f

    lx = 2.       # length of the computational domain
    t_final = 2. # final time of for the computation (assuming t0=0)
    dt = 0.002                   # time step
    nt = int(t_final / dt)       # number of time steps

    nx = 500                     # number of grid points 
    dx = lx / (nx-1)             # grid spacing
    x = np.linspace(0., lx, nx)  # coordinates of grid points
    t = np.linspace(0, t_final, nt+1)
    # initial condition
    u0 = np.exp(-(x-1)**2/0.1)

    # create an array to store the solution
    u = np.empty((nt+1, nx))
    # copy the initial condition in the solution array
    u[0] = u0.copy()
    
    ## make mesh
    X, T = np.meshgrid(x, t[1:])

    for n in range(nt):
        u[n+1] = euler_step(u[n], rhs_backward_periodic, dt, dx, c)

    return t[1:], x, u[1:]

def UQBurgers(c = None):
    def parametric_burgers_rhs(u, t, params):
        k,a,b,c = params
        deriv = a*(1+c*np.sin(t))*u*ifft(1j*k*fft(u)) + b*ifft(-k**2*fft(u))
        return np.real(deriv)

    n = 256
    m = 101
    # Set up grid
    x = np.linspace(-8,8,n+1)[:-1];   dx = x[1]-x[0]
    t = np.linspace(0,10,m);          dt = t[1]-t[0]
    k = 2*np.pi*fftfreq(n, d = dx)
    u0 = np.exp(-(x/2)**2)
    params_dict = {1: (k, c[0], c[1], 0.)}
    u = odeint(parametric_burgers_rhs, u0, t, args=(params_dict[1],)).T

    return t, x, u



