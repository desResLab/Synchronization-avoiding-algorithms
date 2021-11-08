import meshio
import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	print('\n')

Testing_cases = 5
dpi = 300
N_test = 50000
Displacement = np.zeros((N_test,3, Testing_cases))

DNN          = np.zeros((N_test,3, Testing_cases))

Plotter_set  = [-2, 6,12, -3]
plt.rcParams["figure.figsize"] = (20,6)

# steady file
steady_name = "Results/Static/cases="+str(rank)+"/steady.vtk"
steady_file = meshio.read(steady_name)
sd 	  = steady_file.point_data 
sdx_sol = sd['displacement-x']
sdy_sol = sd['displacement-y']
sdz_sol = sd['displacement-z']


for plotter in Plotter_set:
	for i in range(N_test):
		d_name = 'Results/Dynamics/cases='+str(rank) +'/num=' + str(i) + '.vtk'
		d_file = meshio.read(d_name)
		xy    = d_file.points  	  # grid points
		d 	  = d_file.point_data # dic containing all data, by name
		dx_sol = d['displacement-x']
		dy_sol = d['displacement-y']
		dz_sol = d['displacement-z']
		Displacement[i,0],Displacement[i,1],Displacement[i,2] = dx_sol[plotter], dy_sol[plotter], dz_sol[plotter]


	DX,DY,DZ = sdx_sol[plotter], sdy_sol[plotter], sdz_sol[plotter]
	dt = 0.0008534068978707247
	t  = np.linspace(0.0, N_test*dt, N_test)


	fig = plt.figure()
	title_name = 'Point:' + str(xy[plotter])
	fig.suptitle(title_name, fontsize=18)
	plt.subplot(1,3,1)
	plt.plot(t,Displacement[:,0], 'b-' )
	plt.plot(t,DX*np.ones(len(t)), 'k--')
	plt.xlabel('t')
	plt.ylabel('dx')
	plt.subplot(1,3,2)
	plt.plot(t,Displacement[:,1], 'b-' )
	plt.plot(t,DY*np.ones(len(t)), 'k--')
	plt.xlabel('t')
	plt.ylabel('dy')
	plt.subplot(1,3,3)
	plt.plot(t,Displacement[:,2], 'b-' )
	plt.plot(t,DZ*np.ones(len(t)), 'k--')
	plt.xlabel('t')
	plt.ylabel('dz')
	figname = 'IC_Fig/case='+str(rank)+'-P('+str(plotter)+')'+'-end-point-time.png'
	plt.savefig(figname,dpi=dpi)