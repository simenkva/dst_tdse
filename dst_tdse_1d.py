import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst
from icecream import ic

# customize icecream
ic.configureOutput(prefix='')

class DSTSolver:
    def __init__(self, a, b, n):
        self.set_grid(a, b, n)

    def set_grid(self, a, b, n):
        """Set computational grid. Since we are using DST for
        discretization, it is recommended that n = 2^m - 1 for
        some integer m. The grid computed excludes the boundary
        points.
        
        Args:
            a (float): Left boundary.
            b (float): Right boundary.
            n (int): Number of grid points.
        """
        
        ic('Setting up grid.')
        
            
        self.a = a
        self.b = b
        self.n = n
        self.x_full = np.linspace(a, b, n+2)
        self.x = self.x_full[1:-1]
        self.dx = (b - a) / (n+1)
         
        self.k = np.pi / (b - a) * np.arange(1, n+1)

    def set_kinetic_operator(self, T):
        """ Set up kinetic energy operator in terms of
        its diagonal Fourier represenation.
        
        Args:
            T (ndarray or callable): Kinetic energy operator. If callable,
            it is evaluated at the grid points and the result is stored.
        """
        
        # check if T is an ndarray:
        if isinstance(T, np.ndarray):
            self.T = T
        else:
            self.T = T(self.k)

    def set_potential_operator(self, V):
        """ Set up potential energy operator in terms of
        its diagonal space representation.
        
        Args:
            V (ndarray or callable): Potential energy operator. If callable,
            it is evaluated at the grid points and the result is stored.
        """
        
        # check if V is an ndarray:
        if isinstance(V, np.ndarray):
            self.V = V
        else:
            self.V = V(self.x)
            
    def prepare_for_propagation(self, dt):
        """ Prepare the solver for time propagation.
        
        Args:
            dt (float): Time step.
        """
        ic('Preparing for time propagation.')
        self.dt = dt
        self.expV = np.exp(-0.5j * self.dt * self.V)
        self.expT = np.exp(-1.0j * self.dt * self.T)
            
    def propagate(self, psi):
        """ Propagate the wavefunction in time.
        
        Args:
            psi (ndarray): Wavefunction at time t.
            
        Returns:
            ndarray: Wavefunction at time t + dt.
        """
        # compute potential energy term
        psi = self.expV * psi

        # compute kinetic energy term
        psi_hat = dst(psi, type=1)
        psi_hat = self.expT * psi_hat
        psi = idst(psi_hat, type=1)

        # compute potential energy term
        psi = self.expV * psi
        
                
        return psi
        

        
if __name__ == "__main__":
    dt = 0.02
    n_steps = round(2*np.pi / dt) + 1
    ic(n_steps)
    T_fun = lambda k: 0.5 * k**2
    V_fun = lambda x: 0.5 * x**2
    solver = DSTSolver(-5, 5, 255)
    solver.set_kinetic_operator(T_fun)
    solver.set_potential_operator(V_fun)
    solver.prepare_for_propagation(dt)
    
    psi_init = np.exp(-0.5 * (solver.x-2)**2)
    psi_init /= np.linalg.norm(psi_init)
    psi = psi_init.copy()
    psi_hist = np.zeros((n_steps+1, len(psi)), dtype=complex)
    psi_hist[0] = psi_init
    for k in range(n_steps):
        psi = solver.propagate(psi)
        psi_hist[k+1] = psi
        ic(np.linalg.norm(psi))
        
    plt.figure()
    plt.imshow(np.abs(psi_hist)**2, aspect='auto', extent=[solver.a, solver.b, 0, n_steps*dt])
    plt.show()
    
    
    
    