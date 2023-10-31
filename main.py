import numpy as np
import os

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

from algorithms import B_DPP, Falsone, C_SP_SG, DPD_TV

num_of_nodes = 10
num_of_edges = 30 
B = 4 # num of subgraphs
net_dir = f'network/N{num_of_nodes}E{num_of_edges}B{B}'
network = np.load(f'{net_dir}/W_subG.npy')
adj_subG = np.load(f'{net_dir}/adj_subG.npy')

data_dir = 'data/N10_d1_m1'
a = np.load(f'{data_dir}/a.npy')
D = np.load(f'{data_dir}/D.npy')
R = np.load(f'{data_dir}/R.npy')
opt_val = np.loadtxt(f'{data_dir}/opt_val.txt')
opt_val = opt_val.item() # convert the array to a scalar

# compare
C_list = [0.1, 0.8, 1.5, 0.27]
algo_list = [
    B_DPP(network=network, a=a, D=D, R=R, C=C_list[0]), 
    B_DPP(network=network, a=a, D=D, R=R, C=C_list[1]), 
    B_DPP(network=network, a=a, D=D, R=R, C=C_list[2]), 
    B_DPP(network=network, a=a, D=D, R=R, C=C_list[3]), 
    Falsone(network=network, a=a, D=D, R=R, beta=4.4), 
    C_SP_SG(network=network, a=a, D=D, R=R, delta_prime=0.2), 
    DPD_TV(adj_subG=adj_subG, a=a, D=D, R=R, beta=0.1)
]
labels = [
    f'B_DPP_C{C_list[0]}', 
    f'B_DPP_C{C_list[1]}', 
    f'B_DPP_C{C_list[2]}',
    f'B_DPP_C{C_list[3]}',
    'Falsone', 
    'C_SP_SG', 
    'DPD_TV'
]

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

obj_err_list = []
cons_val_list = []
for algo_idx in range(len(algo_list)):
    algo = algo_list[algo_idx]
    logging.info(f'{labels[algo_idx]} is running:')
    obj_err_log = []
    cons_val_log = []
    algo.reset()
    obj_err, cons_val = algo.compute_metrics(opt_val=opt_val)
    obj_err_log.append(obj_err)
    cons_val_log.append(cons_val.item())
    for t in range(1000):
        logging.info(f'ite num {algo.ite_num}')
        algo.step()
        obj_err, cons_val = algo.compute_metrics(opt_val=opt_val)
        obj_err_log.append(obj_err)
        cons_val_log.append(cons_val.item())
    obj_err_list.append(obj_err_log)
    cons_val_list.append(cons_val_log)
    np.savetxt(f'{log_dir}/obj_err_{labels[algo_idx]}.txt', obj_err_log, delimiter=',')
    np.savetxt(f'{log_dir}/cons_val_{labels[algo_idx]}.txt', cons_val_log, delimiter=',')