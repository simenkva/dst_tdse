import numpy as np
from dst_tdse import DSTSolver
from icecream import ic
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

nx = 2047
n_steps = 2048*2
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

ic('Animating ...')

fig = plt.figure(figsize=(8, 8))


# Function to update the plot for each frame
def update(frame):
    plt.clf()
    plt.plot(x, np.abs(psi_hist[frame])**2, 'k')
    plt.text(0.1, 4.5, f'$t = {frame*dt:.2f}$', fontsize=12)
    plt.ylim(0, 5)
    plt.xlim(0, np.pi)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$|psi(x,t)|^2$')
    plt.title('Quantum carpet')
    
# Create the animation
animation = FuncAnimation(fig, update, frames=n_steps+1, interval=1000/30)

# Display the animation
plt.show()


