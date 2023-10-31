import numpy as np
import cvxpy as cp
import os

num_of_nodes = 10
d = 1
m = 1

a = np.random.uniform(low=1., high=2., size=(num_of_nodes, d))
D = np.random.uniform(low=0.5, high=1., size=(num_of_nodes, m, d))
R = np.random.uniform(low=5., high=20., size=m)

save_dir = f'data/N{num_of_nodes}_d{d}_m{m}'
# Save data to corresponding files
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(f'{save_dir}/a.npy', a)
np.save(f'{save_dir}/D.npy', D)
np.save(f'{save_dir}/R.npy', R)

var_x = cp.Variable((num_of_nodes, d))
obj = 0.
for i in range(num_of_nodes):
    obj = obj + 0.5 * cp.quad_form(var_x[i]-a[i], np.identity(d))

coupling_g = np.zeros(m)
for i in range(num_of_nodes):
    coupling_g = coupling_g + D[i] @ var_x[i]
coupling_cons = [coupling_g <= R]
local_cons1 = [var_x[i] >= 0 for i in range(num_of_nodes)]
local_cons2 = [var_x[i] <= 2. for i in range(num_of_nodes)]
cons = coupling_cons + local_cons1 + local_cons2

prob = cp.Problem(cp.Minimize(obj), cons)
prob.solve(solver=cp.MOSEK)
x_star = var_x.value
opt_val = prob.value

np.savetxt(f'{save_dir}/x_star.txt', x_star)
np.savetxt(f'{save_dir}/opt_val.txt', [opt_val])