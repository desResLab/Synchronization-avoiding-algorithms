# extract shared dofs from simulated data
from mpi4py import MPI
import os
import numpy as np
from Tools.commons import *
from Tools.Distributed_tools import *
from numpy import genfromtxt
import h5py
from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	print('\n')
	
# create pathes
path0  = 'Results/sol_on_shared/'
os.makedirs(path0,exist_ok=True)

local_node = (genfromtxt('Results/Rankwised_Data/Rank=' + str(rank) + '_local_nodes.csv', delimiter=',')).astype(int)
shared     = (genfromtxt('Results/Shared_Data/Rank=' + str(rank) + '_shared.csv', delimiter=',')).astype(int)

# local dof
shared_dof = node_to_dof(3, [0,1,2], local_mat_node(shared, local_node))

sol_name   = 'Results/Dynamics/Local-rank-'+str(rank)+'.hdf5' 
	
with h5py.File(sol_name,'r') as f:
	Data_numpy = f['Displacement']
	Data_numpy = np.array(Data_numpy)

	d  = Data_numpy[shared_dof,:] 

hf = h5py.File(path0 + 'rank=' + str(rank) + '-shared_dof.hdf5', 'w')
hf.create_dataset('Displacement', data=d)
hf.close()
