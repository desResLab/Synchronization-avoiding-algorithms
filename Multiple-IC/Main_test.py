# This is a simple 3D linear elasticity solver using classical P1 finite element method
# The clamped plate deformation problem is the classical cantilever beam problem

# we use serial solver now
# This is a set of solvers with different initial conditions, we generate the data by parallel
# We create these initial conditions by imposing several constant uniform forcings, and in the prediction 
# stage, we use a larger force

from Tools.commons import *
from Tools.Mat_construction import *
from Tools.Steady_solvers import *
from Tools.Dynamic_solver import *
from Tools.DNN_prediction import *
import numpy as np
import meshio
import os
from matplotlib import pyplot as plt
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	print('\n')

dpi = 100

# Material properities
E   	= 1e5         # Young's modulus
nu  	= 0.3         # poisson ratio
rho 	= 1           # density

# under-damping + ramped external force
Damp    = 0.5         # mass-proportional damping factor
Ramp    = True        # give a ramped external force
p       = 1           # polynomial order
n_basis = 4           # number of basis function per element
facet_node = 3        # number of facet node
gamma     = .98       # CFL reduction factor

# Mesh info
mesh_name =  "Mesh_info/beam_coarse_p1.vtk"  
Mesh      =  meshio.read(mesh_name)  		# import mesh
Cells     =  (Mesh.cells_dict)['tetra']  
Facets    =  (Mesh.cells_dict)['triangle']
Points    =   Mesh.points                   # nodal points
nELE      =  len(Cells[:,0])    			# number of element
dt   = gamma * Meshsize(Cells, Points)/np.sqrt(E/rho/(1-nu**2)) # permissible time step size

# Dirichlet bc
Neumann = []   # traction free, temporarily 
Dirichlet_node = [] # Clamped Dirichelt BC, x = 0

for i in range(len(Facets)):
	# if this facet is indeed on the boundary x=0 (i.e.: x coordinate = 0)
	if all(abs(Points[Facets[i][k]][0]) < 1e-9 for k in range(facet_node)):
		for j in range(facet_node):
			# then put them into the Dirichlet array, if it is not previously in there
			if Facets[i][j] not in Dirichlet_node:
				Dirichlet_node.append(Facets[i][j])
Dirichlet_global_dof = node_to_dof(3, [0,1,2], Dirichlet_node) # 3D, 3 displacements all 0

fz  	= .4          # external force: body force per unit area
elas = elasticity(E*nu/((1+nu)*(1-2*nu)),E/(2*(1+nu)),rho, fz, Ramp) # elasticity class preparation

# solve for the steady case
if rank == 0:
	print('Start to solving the steady case')
elas_steady = elasticity(E*nu/((1+nu)*(1-2*nu)),E/(2*(1+nu)),rho, fz, False) # elasticity class preparation for the steady solver
d_steady = Steady_Elasticity_solver(p, Cells, Points, Dirichlet_global_dof, elas_steady, t=None, Facets=None, Neumann=None)

dx = d_steady[0::3]
dy = d_steady[1::3]
dz = d_steady[2::3]

# create the folder to save case-wise data
newpath = r'Results/Static'
if not os.path.exists(newpath):
	os.makedirs(newpath)
savename = newpath+'/steady.vtk'
meshio.write_points_cells(savename, Points , Mesh.cells, {'displacement-x':dx, 'displacement-y':dy, 'displacement-z':dz})


test_num  = 20000                # number of time steps
num_cri   = 4000
dnn_every = 1

# create the folder to save case-wise data
newpath = r'Results/Dynamics'
if not os.path.exists(newpath):
	os.makedirs(newpath)

# initialization: d0= v0=0, a0 by solve the system once, d-1 by taylor expansion
d0           =   d_steady
v0           =   np.zeros((len(Points)*3,1)) # initialize v0

M,_,F = Global_Assembly(p,Cells, Points, Dirichlet_global_dof, elas, t=0)
# Taking care of dirichlet BC
for i in range(len(Points)*3):
	for A in [0,1,2]:
		dirich = (node_to_dof(3, [A] , [i]))[0] # global equation number, global dof
		if dirich in Dirichlet_global_dof:
			M[dirich,dirich] = 1
			F[dirich]      = 0
a0 = np.linalg.solve(M,F)

dn           =  d0 - dt*v0 + dt**2/2*a0  # Taylor expansion for the ghost step d_(n-1) 
M_0, _, F_0  =  Global_Assembly_no_bc(p, Cells, Points, elas, 0)	
lumped_M     =  lumping_to_vec(M_0)     # use global assembly to find the lumped mass vector

tn = 0    # current time
if rank == 0:
	print('======Start to solving the unsteady==========')

	# start time integration
for i in range(0,test_num):
	Time = Time_integration_displacement(tn, dt, d0, dn)  # prepare the time integration class
	
	if i <= num_cri - 1:
		d1   = Explicit_assembly(p, Cells, Points, Dirichlet_global_dof, Time, elas, lumped_M, Damp )
		#---------------Warning! dont use elementwise solver now, there is a potential bug there---------------#
		#d1   = Elementwise_explicit_assembly(p, Cells, Points, Dirichlet_global_dof, Time, elas, lumped_M, Damp,F_0)
	else:
		if (i - num_cri)%dnn_every == 0:
			if rank == 0:
				print('Step num is: ' + str(i+1)+ ', current step is being predicted')
			d1 = Dis_prediction(len(Points),Time.d0, Dirichlet_global_dof)
		else: d1 = Explicit_assembly(p, Cells, Points, Dirichlet_global_dof, Time, elas, lumped_M, Damp )
	
	if rank == 0:
		print('current time is :' + str(tn+dt)+ ', step=' + str(i+1) + '/'+str(test_num))
	dn   = d0 
	d0   = d1

	dx = d1[0::3]
	dy = d1[1::3]
	dz = d1[2::3]
	tn = tn+dt
	savename = 'Results/Dynamics/num=' + str(i) + '.vtk'
	meshio.write_points_cells(savename, Points , Mesh.cells, {'displacement-x':dx, 'displacement-y':dy, 'displacement-z':dz})
