import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst, dct, idct

def get_grid(a, b, N, kind='dst', type=1):
    """Generate grid and frequencies for DCT and DST, types 1-4.
    
    Returns the full grid x_all with endpoints if there are some missing from 
    the collocation points x. The frequencies f are also returned. The frequencies
    are such that the transform can be interpreted as mapping between nodal values
    at x[i] and the coefficients of the expansion in terms of the basis functions, 
    which are given by sin(f[j]*x) for DST and cos(f[j]*x) for DCT.
    
    
    
    Args:
        a, b (float): Endpoints of the interval.
        N (int): Number of grid points.
    kind (str): Kind of transform, 'dct' or 'dst'.
    type (int): Type of the transform, 1-4.
        
    Returns:
        x_all (ndarray): All grid points.
        x (ndarray): Collocation points.
        f (ndarray): Frequencies.
    """
    
    
    
    
    
    if kind == 'dst' and type == 1:
        x_all = np.linspace(a, b, N+2)
        x = x_all[1:-1]
        f = np.arange(1, N+1) * np.pi / (b - a)
        return x_all, x, f
    
    if kind == 'dst' and type == 2:
        x_all = np.linspace(a, b, N+1)
        x = x_all[:-1] + (x_all[1] - x_all[0])/2
        f = np.arange(1, N+1) * np.pi / (b-a)
        return x_all, x, f
    
    if kind == 'dst' and type == 3:
        x_all = np.linspace(a, b, N+1)
        x = x_all[1:]
        f = np.arange(1/2, N+1/2) * np.pi / (b - a)
        return x_all, x, f
    
    if kind == 'dst' and type == 4:
        x_all = np.linspace(a, b, N+1)
        x = x_all[:-1] + (x_all[1] - x_all[0])/2
        f = np.arange(1/2, N+1/2) * np.pi / (b - a)
        return x_all, x, f
    
    if kind == 'dct' and type == 1:
        x_all = np.linspace(a, b, N)
        x = x_all
        f = np.arange(0, N) * np.pi / (b-a)
        return x_all, x, f
    
    if kind == 'dct' and type == 2:
        x_all = np.linspace(a, b, N+1)
        x = x_all[:-1] + (x_all[1] - x_all[0])/2
        f = np.arange(0, N) * np.pi / (b-a)
        return x_all, x, f
    
    if kind == 'dct' and type == 3:
        x_all = np.linspace(a, b, N+1)
        x = x_all[:-1]
        f = np.arange(1/2, N+1/2) * np.pi / (b-a)
        return x_all, x, f
    
    if kind == 'dct' and type == 4:
        x_all = np.linspace(a, b, N+1)
        x = x_all[:-1] + (x_all[1] - x_all[0])/2
        f = np.arange(1/2, N+1/2) * np.pi / (b-a)
        return x_all, x, f
    
    raise ValueError('Invalid transform kind or type.')