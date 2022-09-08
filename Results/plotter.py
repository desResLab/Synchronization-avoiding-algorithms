import meshio
import numpy as np
import os
from numpy import genfromtxt
from mpi4py import MPI
from matplotlib import pyplot as plt
import h5py

print('\n')

fs = 16
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)

plt.figure(figsize=(14,5))
# load mesh info
mesh_name =  "../Mesh_info/beam_coarse.vtk"  
Mesh      =  meshio.read(mesh_name)  		# import mesh
Points    =  Mesh.points                    # nodal points

save_every = 1

dt = 0.00024784067462642383 * save_every 

start = int(2000/save_every)  # 350x20
ender = int(1e5/save_every)

for rank in range(2):

	# load exact data
	Path = 'Dynamics/Local-rank-'+str(rank)+'.hdf5'
	hf = h5py.File(Path,'r')
	Data_numpy = np.array(hf['Displacement'][:])
	Data_numpy = Data_numpy.transpose()
	Dis_save = Data_numpy[start:ender,:]


	# load modeled data
	Path = 'Dynamics/Modeled_Local-rank-'+str(rank)+'.hdf5'
	hf = h5py.File(Path,'r')
	Data_numpy = np.array(hf['Displacement'][:])
	Data_numpy = Data_numpy.transpose()
	Dis_save_pred = Data_numpy[start:ender,:]



	# local nodes information, used to identify node coordi
	Rankwise_path = 'Rankwised_Data/Rank='+str(rank)+'_local_nodes.csv'
	rankwise_nodes  =  genfromtxt(Rankwise_path,delimiter=',')
	rankwise_nodes = rankwise_nodes.astype(int)

	
	if rank == 0:
		local_id = 24   # local node num
	if rank == 1:
		local_id = 27

	tracer   = int(3*local_id) # local dof x

	global_node = rankwise_nodes[local_id]
	# global_shared, we dont plot dynamics of a shared node
	g_shared   = (genfromtxt('Shared_Data/Global_shared.csv', delimiter = ',')).astype(int)

	if global_node in list(g_shared):
		print('NOOOOOOOOOOOOO!')
		SOSSOSOSOSOSOSOSSO
	else:
		print('rank='+str(rank)+'-node coordinate is:'+ str(Points[global_node]))


#-----------------------------start to plot-------------------------------------#
	x_range = np.linspace((start+1)*dt, (ender)*dt, num=ender-start)
	bar_    = np.linspace(-100, 100, num=100)
	loc     = ender*0.5*dt
	loc_vec = np.zeros((100,1)) + loc

	alpha = 0.5

	if rank == 0:
		labels = 1
	elif rank == 1:
		labels = 4

	plt.subplot(2,3,labels)
	plt.plot(x_range,Dis_save_pred[:,tracer],'-k', label='Predicted',alpha=alpha, linewidth=2)
	plt.plot(x_range,Dis_save[:,tracer],'--r',label='Truth')
	plt.plot(loc_vec, bar_, '-k', linewidth=0.05)
	plt.xlim([min(x_range), max(x_range)])
	plt.ylim([min(Dis_save_pred[:,tracer]), max(Dis_save_pred[:,tracer])])
	plt.ticklabel_format(style='plain')
	plt.xlabel(r'$t (\textrm{s})$',fontsize=fs)
	plt.ylabel(r'$d_{x} (\textrm{cm})$',fontsize=fs)
	plt.tick_params(labelsize=fs)
	plt.legend(loc='upper right',fontsize=fs-3)

	plt.subplot(2,3,labels+1)
	plt.plot(x_range,Dis_save_pred[:,tracer+1],'-k', label='Predicted',alpha=alpha, linewidth=2)
	plt.plot(x_range,Dis_save[:,tracer+1],'--r',label='Truth')
	plt.plot(loc_vec, bar_, '-k', linewidth=0.05)
	plt.xlim([min(x_range), max(x_range)])
	plt.ylim([min(Dis_save_pred[:,tracer+1]), max(Dis_save_pred[:,tracer+1])])
	plt.ticklabel_format(style='plain')
	plt.xlabel(r'$t (\textrm{s})$',fontsize=fs)
	plt.ylabel(r'$d_{y} (\textrm{cm})$',fontsize=fs)
	plt.tick_params(labelsize=fs)
	plt.legend(loc='upper right',fontsize=fs-3)
	
	plt.subplot(2,3,labels+2)
	plt.plot(x_range,Dis_save_pred[:,tracer+2],'-k',label='Predicted',alpha=alpha, linewidth=2)
	plt.plot(x_range,Dis_save[:,tracer+2],'--r',label='Truth')
	plt.plot(loc_vec, bar_, '-k', linewidth=0.05)
	plt.xlim([min(x_range), max(x_range)])
	plt.ylim([min(Dis_save_pred[:,tracer+2]), max(Dis_save_pred[:,tracer+2])])
	plt.ticklabel_format(style='plain')
	plt.xlabel(r'$t (\textrm{s})$',fontsize=fs)
	plt.ylabel(r'$d_{z} (\textrm{cm})$',fontsize=fs)
	plt.tick_params(labelsize=fs)
	plt.legend(loc='upper right',fontsize=fs-3)
	

plt.tight_layout()
plt.savefig('Comparison.pdf')