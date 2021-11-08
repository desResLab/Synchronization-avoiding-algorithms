import meshio
import numpy as np
from matplotlib import pyplot as plt
import os
# this code will only be used for single partition, in parallel, it is useless
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	print('\n')

dpi    = 100
start  = 4000 # we get rid of the ramping effect, the artificial effect, physically about 3s (3500 steps) to have almost the full loading
N_test = 50000 - start

Dis_save = np.zeros((26,3,N_test))
newpath = r'Dis-training/Data-set/cases='+str(rank) 
if not os.path.exists(newpath):
	os.makedirs(newpath)

for i in range(N_test):
	d_name = "Results/Dynamics_ramped_dataset/cases="+str(rank)+"/num="+str(start + i)+".vtk"
	d_file = meshio.read(d_name)

	d 	   = d_file.point_data # dic containing all data, by name
	dx_sol = d['displacement-x']
	dy_sol = d['displacement-y']
	dz_sol = d['displacement-z']
	
	Dis_save[:,0,i],Dis_save[:,1,i],Dis_save[:,2,i] = dx_sol, dy_sol, dz_sol

	save_name = 'Dis-training/Data-set/cases='+str(rank)+'/num='+str(i)+'.csv'
	np.savetxt(save_name,Dis_save[:,:,i],delimiter=',')