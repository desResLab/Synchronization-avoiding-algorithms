# Distributed 3D linear elasticity solver
from Tools.commons import *
from Tools.Distributed_tools import *
from Tools.Steady_solvers import *
from Tools.Dynamic_solver import *
from Tools.DNN_prediction import *
from mgmetis.parmetis import part_mesh_kway
import numpy as np
import meshio
from math import floor
from mpi4py import MPI
import os
import h5py

device = 'cpu' # in online prediction, use cpu only

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# create pathes
path0  = 'Results/Rankwised_Data/'
os.makedirs(path0,exist_ok=True)

path1  = 'Results/Shared_Data/'
os.makedirs(path1,exist_ok=True)

path2  = 'Results/Static/'
os.makedirs(path2,exist_ok=True)

path3  = 'Results/Dynamics/'
os.makedirs(path3,exist_ok=True)

path4  = 'Results/Rankwised_Element/'
os.makedirs(path4,exist_ok=True)

# Material properities
E   	= 1e6         # Young's modulus
nu  	= 0.3         # poisson ratio
rho 	= 1           # density
fz  	= 0.5         # external force: body force per unit area

# under-damping + ramped external force
Damp    = 0.5         # mass-proportional damping factor
Ramp    = True        # give a ramped external force
p       = 1           # polynomial order
n_basis = 4           # number of basis function per element
facet_node = 3        # number of facet node

elas = elasticity(E*nu/((1+nu)*(1-2*nu)),E/(2*(1+nu)),rho, fz, Ramp) # elasticity class preparation
gamma     = .9                 # CFL reduction factor
test_num  = int(1e5)           # number of test steps
save_every = 1

# parameters from training
nB = 10                      # mini batch size
filter_size = 150            # sample every xx points from the original data set, the $n_s$
learning_rate = 5e-4         # initial learning rate
cut_off     = 0.5            # amount of data used for training
n_future = 20           	 # sequence length of the output
n_past   = 20           	 # sequence length of the input, this lstm is a seq2seq model        
hidden_size = 50
i_cri     = n_past*filter_size - 1         # before this step, all synchronized

#------------------------Step1--------------------------------#
#Get geometry and discretization: gmsh and read with meshio
#Credit: https://pypi.org/project/meshio/

if rank == 0:
	mesh_name =  "Mesh_info/beam_coarse.vtk"  
	Mesh      =  meshio.read(mesh_name)  		# import mesh
	Cells     =  (Mesh.cells_dict)['tetra']  
	Facets    = (Mesh.cells_dict)['triangle']
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
shared_nodes = find_shared_nodes(rank,size,rank_nodal_num,rank_nodal_list) # find the shared nodes for each processor

# define model input size
input_size     = len(shared_nodes)*3

# find scaling constants in training
loc_dof_shared = node_to_dof(3, [0,1,2],local_mat_node(shared_nodes, Local_nodal_list))
data_path = 'Results/sol_on_shared/rank='+str(rank)+'-shared_dof.hdf5'
# use real disaplcement data, full cantilever
X,Y = Dis_data_filtered_subset_coronary(device, input_size, filter_size, n_past, n_future, data_path, cut_off)
_,_,scale_max, scale_min = Scale_to_zero_one(X,Y)

scale_max = scale_max.item()
scale_min = scale_min.item()

# call trained model
model_path = 'Distributed_save/Rank-'+str(rank)+'/nB-'+str(nB)+'-nH-'+str(hidden_size) \
					+'-Lr-'+str(learning_rate)+'-filter='+str(filter_size)+'/model.pth'
model = call_model(device, filter_size, input_size, hidden_size, model_path)  

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
dt       = min(recvbuf2)  # min step size among all procs


#---------------here we call steady solver and gather lumped mass, force and initial data---------------------------#
if rank == 0 :
	print('Start to solving the steady case')
	elas_steady = elasticity(E*nu/((1+nu)*(1-2*nu)),E/(2*(1+nu)),rho, fz, False) # elasticity class preparation for the steady solver
	d_steady = Steady_Elasticity_solver(p, Cells, Points, Dirichlet_global_dof, elas_steady, t=None, Facets=None, Neumann=None)

	dx = d_steady[0::3]
	dy = d_steady[1::3]
	dz = d_steady[2::3]
	meshio.write_points_cells(path2+'steady_distributed.vtk', Points , Mesh.cells, {'displacement-x':dx, 'displacement-y':dy, 'displacement-z':dz})

	# initialization also in the root node
	d0     =   np.zeros((len(Points)*3,1)) # initialize d0
	v0     =   np.zeros((len(Points)*3,1)) # initialize v0

	# calculate lumped mass and pre-assemble the external force (elas_steady is used here since we dont want ramped condition here)
	M_0, _, F_pre    =  Global_Assembly_no_bc(p, Cells, Points, elas_steady, 0)	
	lumped_M         =  lumping_to_vec(M_0)     # use global assembly to find the lumped mass vector

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
	print('Start to solving the unsteady case')

d1_save  = np.zeros((len(Local_nodal_list)*3,int(test_num/save_every)))
tn = 0    # current time
d_sol_shared = np.zeros((test_num, input_size)) # shared node info placeholder

i = 0
counter  = 0 # counter for displacement solution save
counter2 = 0 # counter for shift the prediction

# start time integration
while i<test_num:

	if i <= i_cri:  	# do syn steps 
		Time = Time_integration_displacement(tn, dt, d_0, d_n)  # prepare the time integration class
		
		# call parallel explicit solver with pre-assembly and synchronization
		d1 = parallel_explicit_solver_dis_pre(LocalK, F_rankwise, Points, Local_nodal_list, Local_Dirichlet,\
									Time, elas, l_M, Damp, size, rank, MODEL=False)
		
		d_sol_shared[i,:] = d1[loc_dof_shared,0] # save for refilling
		
		if rank == 0:
			print('current time is :' + str(tn+dt)+ ', step=' + str(i+1) + '/'+str(test_num))

		# save if matched
		if i % save_every == 0:
			d1_save[:,counter] = d1.reshape((len(d1)))
			counter += 1


		# update solutions
		d_n   = d_0 
		d_0   = d1
		tn = tn+dt	
		i  = i + 1

	else: # start to use the data-driven model to avoid sync

		# call model predictor
		d_shared = encoder_decoder_predictor(device, i, model,n_past, n_future, filter_size, input_size,\
												d_sol_shared, scale_max, scale_min)

		# start refilling steps
		for k in range(i,i+n_future*filter_size):
			
			if k >= test_num: # if exceeds total steps, break
				break 

			else: # do displacement update without sync and prepare for refilling

				Time = Time_integration_displacement(tn, dt, d_0, d_n)  # prepare the time integration class

				# call parallel explicit solver with pre-assembly and without synchronization
				d1 = parallel_explicit_solver_dis_pre(LocalK, F_rankwise, Points, Local_nodal_list, Local_Dirichlet,\
									Time, elas, l_M, Damp, size, rank, MODEL=True) # no syn
				
				# update values on the shared nodes by predicted values
				d1[loc_dof_shared] = d_shared[k-i_cri-1-n_future*filter_size*counter2,:].reshape((input_size,1))

				# save for refilling
				d_sol_shared[i,:] = d1[loc_dof_shared,0]
				
				if rank == 0:
					print('current time is :' + str(tn+dt)+ ', step=' + str(i+1) + '/'+str(test_num))

				# save if matched
				if i % save_every == 0:
					d1_save[:,counter] = d1.reshape((len(d1)))
					counter += 1


				# update solutions
				d_n   = d_0
				d_0   = d1
				i  = i+1
				tn = tn + dt

		counter2 += 1

# save
save_name = path3 + 'Modeled_Local-rank-'+str(rank)+'.hdf5'
hf = h5py.File(save_name, 'w')
hf.create_dataset('Displacement', data=d1_save, compression='gzip')
hf.close()
