import meshio
import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	print('\n')

Testing_cases = 1
dpi = 100
N_test = 20000
Displacement = np.zeros((N_test,3))

DNN          = np.zeros((N_test,3))

Plotter_set  = [-2, 6,12, -3]
plt.rcParams["figure.figsize"] = (20,6)

# steady file
steady_name = "Results/Static/steady.vtk"
steady_file = meshio.read(steady_name)
sd 	  = steady_file.point_data 
sdx_sol = sd['displacement-x']
sdy_sol = sd['displacement-y']
sdz_sol = sd['displacement-z']


for plotter in Plotter_set:
	for i in range(N_test):
		d_name = 'Results/Dynamics_truth/num=' + str(i) + '.vtk'
		d_file = meshio.read(d_name)
		xy    = d_file.points  	  # grid points
		d 	  = d_file.point_data # dic containing all data, by name
		dx_sol = d['displacement-x']
		dy_sol = d['displacement-y']
		dz_sol = d['displacement-z']
		Displacement[i,0],Displacement[i,1],Displacement[i,2] = dx_sol[plotter], dy_sol[plotter], dz_sol[plotter]


		d_name2 = 'Results/Dynamics/num=' + str(i) + '.vtk'
		d_file2 = meshio.read(d_name2)
		d2 	  = d_file2.point_data # dic containing all data, by name
		dx_sol2 = d2['displacement-x']
		dy_sol2 = d2['displacement-y']
		dz_sol2 = d2['displacement-z']
		DNN[i,0],DNN[i,1],DNN[i,2] = dx_sol2[plotter], dy_sol2[plotter], dz_sol2[plotter]




	DX,DY,DZ = sdx_sol[plotter], sdy_sol[plotter], sdz_sol[plotter]
	dt = 0.0008534068978707247
	t  = np.linspace(0.0, N_test*dt, N_test)


	fig = plt.figure()
	title_name = 'Point:' + str(xy[plotter])
	fig.suptitle(title_name, fontsize=18)
	plt.subplot(1,3,1)
	plt.plot(t,Displacement[:,0], 'b-' )
	plt.plot(t,DNN[:,0], 'r-' )
	plt.plot(t,DX*np.ones(len(t)), 'k--')
	plt.xlabel('t')
	plt.ylabel('dx')
	plt.subplot(1,3,2)
	plt.plot(t,Displacement[:,1], 'b-' )
	plt.plot(t,DNN[:,1], 'r-' )
	plt.plot(t,DY*np.ones(len(t)), 'k--')
	plt.xlabel('t')
	plt.ylabel('dy')
	plt.subplot(1,3,3)
	plt.plot(t,Displacement[:,2], 'b-' )
	plt.plot(t,DNN[:,2], 'r-' )
	plt.plot(t,DZ*np.ones(len(t)), 'k--')
	plt.xlabel('t')
	plt.ylabel('dz')
	figname = 'Fig/case='+str(rank)+'-P('+str(plotter)+')'+'-end-point-time.png'
	plt.savefig(figname,dpi=dpi)