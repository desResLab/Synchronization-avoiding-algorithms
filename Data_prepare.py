from Tools.commons import *
from Tools.Distributed_tools import *
from Tools.Steady_solvers import *
from Tools.Dynamic_solver import *
from mgmetis.parmetis import part_mesh_kway
import numpy as np
import meshio
from math import floor
from mpi4py import MPI
import os
import h5py

# MPI spec.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# create pathes
path0  = 'Results/Rankwised_Data/' # save rankwise node labels
os.makedirs(path0,exist_ok=True)

path1  = 'Results/Shared_Data/'    # save Shared node labels
os.makedirs(path1,exist_ok=True)

path2  = 'Results/Static/'         # save steady solution d = K^{-1} F, if solved
os.makedirs(path2,exist_ok=True)

path3  = 'Results/Dynamics/'       # save dynamic solutions by explicit method
os.makedirs(path3,exist_ok=True)

path4  = 'Results/Rankwised_Element/' # save rankwise element labels
os.makedirs(path4,exist_ok=True)

# Material properities
E   	= 1e6         # Young's modulus
nu  	= 0.3         # poisson ratio
rho 	= 1           # density
fz  	= 0.5         # external force: body force per unit volume

# under-damping + ramped external force
Damp    = 0.5         # mass-proportional damping factor
Ramp    = True        # give a ramped external force
p       = 1           # polynomial order, note p=2 only works for steady case, dynamic case requires advanced lumping method
n_basis = 4           # number of basis function per element
facet_node = 3        # number of facet node

elas = elasticity(E*nu/((1+nu)*(1-2*nu)),E/(2*(1+nu)),rho, fz, Ramp) # elasticity class preparation
gamma     = .9                 # CFL reduction factor
test_num  = int(1e5)           # number of total test steps
save_every = 1                 # save dynamic solution for every i steps

#------------------------Step1--------------------------------#
#Get geometry and discretization using gmsh and read with meshio
#Credit: https://pypi.org/project/meshio/

if rank == 0:
	mesh_name =  "Mesh_info/beam_coarse.vtk"  
	Mesh      =  meshio.read(mesh_name)  		# import mesh
	Cells     =  (Mesh.cells_dict)['tetra']     # tetra element labels
	Facets    = (Mesh.cells_dict)['triangle']   # tri facet labels
	Points    =  Mesh.points                    # nodal points
	nELE      =  len(Cells[:,0])    			# number of element

	# find the elmdist (vtxdist) array: 
	# credit: meshPartition.py from the CVFES repo:https://github.com/desResLab/CVFES
	nEach = floor(nELE/size)      # average number of element to each P
	nLeft = nELE - nEach * size   # leftovers evenly distributed to the last a few Ps
	elmdist = np.append( (nEach + 1) * np.arange(nLeft +1), \
									( (nEach+1)*nLeft) + nEach * np.arange(1, size -nLeft +1))

	elmdist = elmdist.astype(np.int64) # convert to integers
else:
	Cells,Facets,Points,elmdist = None, None, None, None # placeholders

# broadcast the data to each processor
Cells   = comm.bcast(Cells,   root=0)
Facets  = comm.bcast(Facets,  root=0)
Points  = comm.bcast(Points,  root=0)
elmdist = comm.bcast(elmdist, root=0)

# allocate element nodes to each processors, i.e.: find the array eptr, eind
P_start   = elmdist[rank]
P_end     = elmdist[rank+1]

eptr = np.zeros(P_end-P_start+1, dtype=np.int64)
eind = np.empty(0, dtype=np.int64)

for idx, ele in enumerate(Cells[P_start:P_end]):
	eptr[idx+1] = eptr[idx] + len(ele)
	eind = np.append(eind, ele[:])	

# using mgmetis.parmetis for mesh partitioning:
# Credit: https://github.com/chiao45/mgmetis
_, epart = part_mesh_kway(size, eptr, eind)

# gather the partitioned data, use Gatherv function to concatenate partitioned array of different size
recvbuf = None
if rank == 0:
    recvbuf = np.empty(len(Cells), dtype='int')
comm.Gatherv(epart,recvbuf,root=0)
recvbuf = comm.bcast(recvbuf, root=0) 

# gather local element and node list
Local_ele_list, Local_nodal_list = rankwise_dist(rank, recvbuf, Points, Cells)

# Collect the shared nodes information
rank_nodal_num  = comm.gather(len(Local_nodal_list),root=0) 
rank_nodal_list = comm.gather(Local_nodal_list,root=0)
rank_nodal_num,rank_nodal_list = comm.bcast(rank_nodal_num,root=0), comm.bcast(rank_nodal_list, root=0)

# find the shared nodes for each processor
shared_nodes = find_shared_nodes(rank,size,rank_nodal_num,rank_nodal_list) 


# save shared nodes information
np.savetxt(path1+'Rank='+str(rank)+'_shared.csv',shared_nodes,delimiter=',',fmt='%d')
np.savetxt(path0+'Rank='+str(rank)+'_local_nodes.csv',rank_nodal_list[rank],delimiter=',',fmt='%d')
np.savetxt(path4+'Rank='+str(rank)+'_elements.csv',Local_ele_list,delimiter=',',fmt='%d')

# gather global shared node information
G_shared_nodes = comm.gather(shared_nodes,root=0)
if rank == 0:
	Global_shared = sort_shared(G_shared_nodes)
	np.savetxt(path1+'Global_shared.csv',Global_shared,delimiter=',',fmt='%d')

# get Localized Dirichlet array
Dirichlet_node = [] # Clamped Dirichelt BC, x = 0
if rank == 0:
	for i in range(len(Facets)):
		# if this facet is indeed on the boundary x=0 (i.e.: x coordinate = 0)
		if all(abs(Points[Facets[i][k]][0]) < 1e-9 for k in range(facet_node)):
			for j in range(facet_node):
				# then put them into the Dirichlet array, if it is not previously in there
				if Facets[i][j] not in Dirichlet_node:
					Dirichlet_node.append(Facets[i][j])
	Dirichlet_global_dof = node_to_dof(3, [0,1,2], Dirichlet_node) # 3D, 3 displacements all 0


else:
	Dirichlet_node = None # placeholder
Dirichlet_node = comm.bcast(Dirichlet_node, root=0) # Global Dirichlet node, broadcasted to all processors

# processor-wised dirichlet bc
Local_Dirichlet 	= Dirichlet_rank_dist(Dirichlet_node, Local_nodal_list) 

# find the time step size
dt   = gamma * Meshsize(Cells[Local_ele_list,:], Points)/np.sqrt(E/rho/(1-nu**2)) 
# gather dt from each processor and pick the minimal one for time integration
recvbuf2 = None
if rank == 0:
    recvbuf2 = np.empty(size, dtype='float')
comm.Gather(dt,recvbuf2,root=0)
recvbuf2 = comm.bcast(recvbuf2, root=0)
dt       = min(recvbuf2) # min step size among all procs


#---------------here we call steady solver and gather lumped mass, force and initial data---------------------------#
if rank == 0 :
	print('Start to solving the steady case')
	# elasticity class preparation for the steady solver
	elas_steady = elasticity(E*nu/((1+nu)*(1-2*nu)),E/(2*(1+nu)),rho, fz, False) 
	# solve steady solutions
	d_steady = Steady_Elasticity_solver(p, Cells, Points, Dirichlet_global_dof, elas_steady, t=None, Facets=None, Neumann=None)
	# save steady solutions
	dx = d_steady[0::3]
	dy = d_steady[1::3]
	dz = d_steady[2::3]
	meshio.write_points_cells(path2+'steady_distributed.vtk', Points , Mesh.cells, {'displacement-x':dx, 'displacement-y':dy, 'displacement-z':dz})

	# initialization also in the root node
	d0     =   np.zeros((len(Points)*3,1)) # initialize d0
	v0     =   np.zeros((len(Points)*3,1)) # initialize v0

	# calculate lumped mass and pre-assemble the external force (elas_steady is used here since we dont want ramped condition here)
	M_0, _, F_pre  =  Global_Assembly_no_bc(p, Cells, Points, elas_steady, 0)	
	lumped_M     =  lumping_to_vec(M_0)     # use global assembly to find the lumped mass vector

	# calculate the ghost step dn. However, if ramped condition is considered, dn is just zero since F(t=0) is zero
	M,K,F = Global_Assembly(p,Cells, Points, Dirichlet_global_dof, elas, t=0)
	# Taking care of dirichlet BC
	for i in range(len(Points)*3):
		for A in [0,1,2]:
			dirich = (node_to_dof(3, [A] , [i]))[0] # global equation number, global dof
			if dirich in Dirichlet_global_dof:
				M[dirich,dirich] = 1
				F[dirich]      = 0
	a0 = np.linalg.solve(M,F-K@d0)
	dn           =  d0 - dt*v0 + dt**2/2*a0  # Taylor expansion for the ghost step d_(n-1) 
	dn           = dn.reshape((len(Points)*3,1))
else:
	lumped_M, d0, dn, F_pre = None,None,None,None

# broadcast to each processor
lumped_M  = comm.bcast(lumped_M,  root=0)
d0  	  = comm.bcast(d0,  root=0)
dn  	  = comm.bcast(dn,  root=0)
F_pre     = comm.bcast(F_pre, root=0)

# Localize "synchronized" quantities
local_dof   = node_to_dof(3,[0,1,2],Local_nodal_list)
F_rankwise  = F_pre[local_dof] 
l_M = lumped_M[local_dof] 
d_0 = d0[local_dof]
d_n = dn[local_dof]

# pre-assemble the stiffness matrix
Local_cell = Cells[Local_ele_list,:]
LocalK = Local_assembly_for_stiffness(Local_nodal_list, \
					Local_cell, Points, p, n_basis, elas, rank)

# start to solve the unsteady problem
if rank == 0:
	print("Time-step size is: " + str(dt))

tn = 0    # current time
if rank == 0:
	print('Start to solving the unsteady case')

d1_save  = np.zeros((len(Local_nodal_list)*3,int(test_num/save_every)))
counter  = 0

# start time integration
for i in range(0,test_num):
	Time = Time_integration_displacement(tn, dt, d_0, d_n)  # prepare the time integration class
	
	# call parallel explicit solver with pre-assembly
	d1 = parallel_explicit_solver_dis_pre(LocalK, F_rankwise, Points, Local_nodal_list, Local_Dirichlet,\
									Time, elas, l_M, Damp, size, rank, MODEL=False)
	if rank == 0:
		print('current time is :' + str(tn+dt)+ ', step=' + str(i+1) + '/'+str(test_num))

	# update solutions
	d_n   = d_0 
	d_0   = d1
	tn = tn+dt	
	
	# save every xx steps
	if i % save_every == 0:
		d1_save[:,counter] = d1.reshape((len(d1)))
		counter += 1

# save as hdf5 format
save_name = path3 + 'Local-rank-'+str(rank)+'.hdf5'
hf = h5py.File(save_name, 'w')
hf.create_dataset('Displacement', data=d1_save, compression='gzip')
hf.close()
