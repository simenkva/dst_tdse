import numpy as np
from dst_tdse import DSTSolver
from icecream import ic
import matplotlib.pyplot as plt

nx = 2047
n_steps = 2048
dt = np.pi/n_steps
solver = DSTSolver(0.0, np.pi, nx)
x = solver.xx[0]
solver.set_kinetic_operator(lambda k: 0.5 * k**2)
solver.set_potential_operator(lambda x: 0.0)
solver.prepare_for_propagation(dt)

psi = np.ones(nx, dtype=complex)

    
psi_hist = np.zeros((n_steps+1, nx), dtype=complex)
psi_hist[0] = psi

ic('Propagating...  ')
for k in range(n_steps):
    psi = solver.propagate(psi)
    psi_hist[k+1] = psi

plt.figure(figsize=(8,8))
plt.imshow(np.abs(psi_hist)**2, 
           aspect='auto', origin='lower',
           extent=[0, np.pi, 0, n_steps*dt], 
           cmap='hot')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Quantum carpet')
plt.show()
