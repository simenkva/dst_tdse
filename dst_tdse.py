import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dstn, idstn
from icecream import ic
from dct_dst import get_grid

# customize icecream
ic.configureOutput(prefix='')


class DSTSolver:
    """Solver for time-dependent Schrödinger equation (TDSE) using
    Discrete Sine Transform (DST) for spatial discretization.
    
    The solver solves the TDSE on the form
    $$ i \partial_t \psi(t) = (T + V + f(t)D) \psi(t) $$
    where $T$ is the kinetic energy operator, $V$ is a time-independent
    potential and $D$ is a potential modulated by $f(t)$.
    
    $T$ is given as any operator diagonal in  Fourier space, $V$ and $D$
    are given as any operator diagonal in position space. They are set as callable functions]
    or as ndarrays. If they are set as callables, they are evaluated at the grid points.
    """
    
    def __init__(self, a, b, n):
        """ Initialize the solver with a grid. See set_grid."""
        self.set_grid(a, b, n)

    def set_grid(self, a, b, n):
        """Set computational grid. Since we are using DST for
        discretization, it is recommended that n = 2^m - 1 for
        some integer m. The grid computed excludes the boundary
        points.
        
        If set_grid is called with a, b and n as arrays, the
        grid will be set up as a multidimensional grid. If scalars are
        given, the grid will be set up as a one-dimensional grid.
        
        Args:
            a (float or ndarray): Left boundary/boundaries.
            b (float or ndarray): Right boundary/boundaries.
            n (int or ndarray of ints): Number of grid points.
        """
        
        ic('Setting up grid.')
        
            
        self.a = np.atleast_1d(a)
        self.b = np.atleast_1d(b)
        self.n = np.atleast_1d(n)
        assert(len(self.a) == len(self.b))
        assert(len(self.a) == len(self.n))
        self.dim = len(self.a)
        self.x_full = []
        self.x = []
        self.dx = []
        self.k = []
        for i in range(self.dim):
            a = self.a[i]
            b = self.b[i]
            n = self.n[i]
            x_full, x, k = get_grid(a, b, n, kind='dst', type=1)
            self.x_full.append(x_full)
            self.x.append(x)
            self.dx.append(x[1] - x[0])
            self.k.append(k)
            # self.x_full.append(np.linspace(a, b, n+2))
            # self.x.append( self.x_full[i][1:-1] )
            # self.dx.append( (b - a) / (n+1) )
            # self.k.append( np.pi / (b - a) * np.arange(1, n+1) )

        self.xx = np.meshgrid(*self.x, indexing='ij')
        self.kk = np.meshgrid(*self.k, indexing='ij')

    def set_kinetic_operator(self, T):
        """ Set up kinetic energy operator in terms of
        its diagonal Fourier represenation.
        
        Args:
            T (ndarray or callable): Kinetic energy operator. If callable,
            it is evaluated at the grid points and the result is stored.
        """
        
        # check if T is an ndarray:
        if isinstance(T, np.ndarray):
            assert(T.shape == self.n)
            self.T = T
        else:
            self.T = T(*self.kk)

    def set_td_modulation_function(self, f):
        """ Set up time-dependent potential modulation function. See also
        set_td_potential_operator.
        
        Args:
            f (callable): Time-dependent modulation function.
        """
        self.f = f
        
    def set_td_potential_operator(self, D):
        """ Set up time-dependent potential energy operator in terms of
        its diagonal space representation. Remember to also set the
        time-dependent potential modulation function.
        
        Args:
            D (ndarray or callable): Potential energy operator. If callable,
            it is evaluated at the grid points and the result is stored.
        """
        
        # check if D is an ndarray:
        if isinstance(D, np.ndarray):
            assert(D.shape == self.n)
            self.D = D
        else:
            self.D = D(*self.xx)
            
    def set_potential_operator(self, V):
        """ Set up potential energy operator in terms of
        its diagonal space representation.
        
        Args:
            V (ndarray or callable): Potential energy operator. If callable,
            it is evaluated at the grid points and the result is stored.
        """
        
        # check if V is an ndarray:
        if isinstance(V, np.ndarray):
            assert(V.shape == self.n)
            self.V = V
        else:
            self.V = V(*self.xx)
        
    def get_energy(self, psi):
        """ Compute the energy of the wavefunction.
        
        Args:
            psi (ndarray): Wavefunction.
            
        Returns:
            float: Energy of the wavefunction (ignoring time-dependent potential)
        """
        psi_hat = dstn(psi, type=1)
        Tpsi_hat = self.T * psi_hat
        Tpsi = idstn(Tpsi_hat, type=1)
        Vpsi = self.V * psi
        E = np.sum(np.conj(psi) * (Tpsi + Vpsi)).real / np.sum(np.conj(psi) * psi)
        return E
            
    def prepare_for_propagation(self, dt):
        """ Prepare the solver for time propagation.
        
        Args:
            dt (float): Time step.
        """
        ic('Preparing for time propagation.')
        self.dt = dt
        self.expV = np.exp(-0.5j * self.dt * self.V)
        self.expT = np.exp(-1.0j * self.dt * self.T)
        
            
    def propagate(self, psi, t=0.0):
        """ Propagate the wavefunction from time t to time t + dt.
        The algorithm used is a split-step propagation method,
        
        $$ psi(t + 3dt/4) = U_V(t) U_T U_V(t+h/4) psi(t) $$
        
        where
        
        $$ U_T = exp(-i dt T) $$
        $$ U_V(t) = exp(-i dt V(t)/2), \quad V(t) = V + f(t)D $$
    
        This scheme has local error $O(dt^3)$.
        
        Args:
            psi (ndarray): Wavefunction at time t.
            
        Returns:
            ndarray: Wavefunction at time t + dt.
        """
        
        assert(np.all(psi.shape == self.n))
        
        # compute time-dependent potential energy term
        if hasattr(self, 'f') and hasattr(self, 'D'):
            self.expD = np.exp(-0.5j * self.dt * self.D * self.f(t + 0.25*self.dt))
            psi = self.expD * psi
            
        # compute potential energy term
        psi = self.expV * psi

        # compute kinetic energy term
        psi_hat = dstn(psi, type=1)
        psi_hat = self.expT * psi_hat
        psi = idstn(psi_hat, type=1)

        # compute potential energy term
        psi = self.expV * psi

        # compute time-dependent potential energy term
        if hasattr(self, 'f') and hasattr(self, 'D'):
            self.expD = np.exp(-0.5j * self.dt * self.D * self.f(t + 0.75*self.dt))
            psi = self.expD * psi

        return psi
        

        
def test_1d():
    dt = 0.02
    n_steps = round(2*np.pi / dt) + 1
    ic(n_steps)
    T_fun = lambda k: 0.5 * k**2
    V_fun = lambda x: 0.5 * x**2
    solver = DSTSolver(-5, 5, 15)
    solver.set_kinetic_operator(T_fun)
    solver.set_potential_operator(V_fun)
    solver.prepare_for_propagation(dt)
    ic(solver.V)
    ic(solver.T)
    
    ic(solver.x[0], solver.k[0 ])
    psi_init = np.exp(-0.5 * (solver.x[0]-2)**2)
    psi_init /= np.linalg.norm(psi_init)
    psi = psi_init.copy()
    psi_hist = np.zeros((n_steps+1, len(psi)), dtype=complex)
    psi_hist[0] = psi_init
    for k in range(n_steps):
        psi = solver.propagate(psi)
        psi_hist[k+1] = psi
        ic(np.linalg.norm(psi))
        
    plt.figure()
    plt.imshow(np.abs(psi_hist)**2, aspect='auto', extent=[solver.a[0], solver.b[0], 0, n_steps*dt])
    plt.show()
    
    
def test_2d():
    dt = 0.02
    t_final = 100
    n_steps = round(t_final / dt) + 1
    ic(n_steps)
    t_range = np.linspace(0, n_steps*dt, n_steps+1)

    T_fun = lambda kx, ky: 0.5 * kx**2 + 0.5*ky**2
    V_fun = lambda x, y: -1.0 * np.exp(- x**2 -y**2 )
    D_fun = lambda x, y: x
    f_fun = lambda t: 0.05 * np.cos(0.057*t) * np.sin(np.pi*t/100)**2
    solver = DSTSolver([-100, -100], [100, 100], [255, 255])
    solver.set_kinetic_operator(T_fun)
    solver.set_potential_operator(V_fun)
    solver.set_td_potential_operator(D_fun)
    solver.set_td_modulation_function(f_fun)

    ic('Computing initial wavefunction by imaginary time propagation')
    solver.prepare_for_propagation(-1j*dt)
    psi = np.exp(-solver.xx[0]**2-solver.xx[1]**2)
    psi /= np.linalg.norm(psi)
    for k in range(10000):
        psi_new = solver.propagate(psi)
        psi_new /= np.linalg.norm(psi_new)
        delta = np.linalg.norm(psi - psi_new)
        psi = psi_new
        ic(solver.get_energy(psi), delta)
        if delta < 1e-6:
            ic('Energy sufficiently converged!')
            break
            
            
    plt.figure()
    plt.imshow(np.abs(psi), extent=[solver.a[0], solver.b[0], solver.a[1], solver.b[1]], origin='lower'
               , aspect='auto')
    plt.colorbar()
    plt.title('Initial wavefunction/ground state')
    plt.show()
    
    vmax = np.max(np.abs(psi))
    
    solver.prepare_for_propagation(dt)
    for k, t in enumerate(t_range):
        psi = solver.propagate(psi, t)
        #ic(np.linalg.norm(psi))
        if k % 500 == 0:
            ic('Plotting wavefunction at t =', t)
            plt.figure()
            plt.imshow(psi.real, vmin =-vmax,
                       vmax = vmax,
                       extent=[solver.a[0], solver.b[0], solver.a[1], solver.b[1]], origin='lower'
                       , aspect='auto', cmap='seismic')
            plt.colorbar()
            plt.title(f'Wavefunction at t = {t}')

    plt.show()
    
    
    
def test_3d():
    dt = 0.1
    t_final = 100
    n_steps = round(t_final / dt) + 1
    ic(n_steps)
    t_range = np.linspace(0, n_steps*dt, n_steps+1)

    T_fun = lambda kx, ky, kz: 0.5 * kx**2 + 0.5*ky**2 + 0.5*kz**2
    V_fun = lambda x, y, z: -1.0 * (0.01 + x*x+y*y+z*z)**(-.5)
    L = 15
    solver = DSTSolver([-L, -L, -L], [L, L, L], [127, 127, 127])
    solver.set_kinetic_operator(T_fun)
    solver.set_potential_operator(V_fun)

    ic('Computing initial wavefunction by imaginary time propagation')
    solver.prepare_for_propagation(-1j*dt)
    r2 = solver.xx[0]**2 + solver.xx[1]**2 + solver.xx[2]**2
    psi = solver.xx[1]*np.exp(-r2**.5)
    psi /= np.linalg.norm(psi)
    for k in range(10000):
        psi_new = solver.propagate(psi)
        psi_new /= np.linalg.norm(psi_new)
        delta = np.linalg.norm(psi - psi_new)
        psi = psi_new
        ic(solver.get_energy(psi), delta)
        if delta < 1e-6:
            ic('Energy sufficiently converged!')
            break
            
            
    plt.figure()
    bmp = psi[:, :, 64].squeeze()
    plt.imshow(np.abs(bmp), extent=[solver.a[0], solver.b[0], solver.a[1], solver.b[1]], origin='lower'
               , aspect='auto')
    plt.colorbar()
    plt.title('Eigenfunction of approximate Coulomb potential.')
    plt.show()
    
    
if __name__ == "__main__":
    test_1d()
    #test_3d()
    
    
    
    
    