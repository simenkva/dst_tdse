import numpy as np
from dst_tdse import DSTSolver
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

x_min = -20.0
x_max = 20.0
nx = 4095
t_final = 10
n_steps = t_final * 1000
animate_every = 10
t_range = np.linspace(0, t_final, n_steps+1)
dt = t_range[1] - t_range[0]
print(t_range[1]-t_range[0])

solver = DSTSolver(x_min, x_max, nx)
x = solver.xx[0]
solver.set_kinetic_operator(lambda k: 0.5 * k**2)
def potential(x):
    if x > 0:
        return 20
    else:
        return 0
    
potential = lambda x: 0.5*x**2
    
solver.set_potential_operator(np.vectorize(potential))
solver.prepare_for_propagation(dt)

psi = np.exp(-0.5*(x+10)**2 + 5j*x)

             
psi_hist = np.zeros((n_steps+1, nx), dtype=complex)
psi_hist[0] = psi

print('Propagating...  ')
for k in range(n_steps):
    psi = solver.propagate(psi)
    psi_hist[k+1] = psi

print('Animating ...')

fig = plt.figure(figsize=(8, 8))


# Function to update the plot for each frame
def update(frame):
    plt.clf()
    plt.plot(x, np.abs(psi_hist[frame])**2, 'k')
    plt.text(0.1, 4.5, f'$t = {frame*dt:.2f}$', fontsize=12)
    plt.ylim(0, 5)
    plt.xlim(x_min, x_max)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$|psi(x,t)|^2$')
    plt.title('Quantum carpet')
    
# Create the animation
animation = FuncAnimation(fig, update, frames=range(0, n_steps+1, animate_every), interval=1000/30)

# Display the animation
plt.show()


