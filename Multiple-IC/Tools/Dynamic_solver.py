import numpy as np
from Tools.Mat_construction import *
from Tools.commons import *
from Tools.Distributed_tools import *

#from Tools.DNN_prediction import *

# the following solvers are all displacement based

# solve dynamics by global assembly
def Explicit_assembly(p,Cells, Points, Dirichlet, T, Elas, lumped_M, alpha):
	
	_,K,F = Global_Assembly_no_bc(p, Cells, Points, Elas, T.tn)

	F = F.reshape((len(F),1))
	
	d1 = T.dt**2*F - (T.dt**2*K) @ (T.d0) + 2*lumped_M * (T.d0) - \
				lumped_M*(T.dn) + T.dt/2*alpha*lumped_M*(T.dn)
	d1=d1/(lumped_M + 0.5*alpha*lumped_M*T.dt)

	for i in range(len(d1)):
		if i in Dirichlet:
			d1[i] = 0 
	return d1

# solve dynamics by elementwise operation
def Elementwise_explicit_assembly(p, Cells, Points, Dirichlet, T, Elas, lumped_M, alpha, F_0):
	n_basis = 4
	d1  = np.zeros((len(Points)*3,1)) 
	F_int 	 = np.zeros((len(Points)*3,1))  # dof-wise internal force allocation
	F_ext    = np.zeros((len(Points)*3,1))  # dof-wise external force allocation

	# loop over all elemenet
	for i in range(len(Cells)):
		ele = Cells[i] # locate local nodes
		point = []
		for n_b in range(n_basis):
			point.append(Points[ele[n_b]])

		P     = np.array(point)
		_, Ke ,_ 	  = Local_MKF(p, n_basis, P, Elas, None, None, T.tn) 

		# find the corresponding slices of the computed displacement vector
		ele_dof      = node_to_dof(3,[0,1,2],ele)  
		F_int[ele_dof] += Ke @ (T.d0)[ele_dof]


	d1  = (T.dt**2*(F_0 - F_int) + 2*lumped_M*T.d0 - lumped_M*T.dn + T.dt/2*lumped_M*alpha*T.dn)/(lumped_M + 0.5*alpha*lumped_M*T.dt) 


	# strong enforce the dirichlet bc
	for i in range(len(d1)):
		if i in Dirichlet:
			d1[i] = 0 

	return d1

# solve dynamics by parallel solver
def parallel_explicit_solver_dis(p, Cells, Points, Local_nodes, Local_Dirichlet,\
									T, Elas, l_M, alpha, f_0, size, rank, comm, num, num_cri,\
									dnn_every, shared_nodes):
	
	n_basis = 4
	d1       = np.zeros((len(Local_nodes)*3,1))  # dof-wise displacement allocation
	F_int 	 = np.zeros((len(Local_nodes)*3,1))  # dof-wise internal force allocation
	F_ext    = np.zeros((len(Local_nodes)*3,1))  # dof-wise external force allocation

	# loop over all elemenet
	for i in range(len(Cells)):
		ele = Cells[i] # locate local nodes
		point = []
		for n_b in range(n_basis):
			point.append(Points[ele[n_b]])

		P     = np.array(point)
		_, Ke ,_ 	  = Local_MKF(p, n_basis, P, Elas, None, None, T.tn) 

		# find the corresponding slices of the computed displacement vector
		ele_local    = local_mat_node(ele,Local_nodes)   
		ele_dof      = node_to_dof(3,[0,1,2],ele_local)  

		F_int[ele_dof] += Ke @ (T.d0)[ele_dof]

	# compute the unsynchronized value, to be used for dnn
	d1  = (T.dt**2*(f_0 - F_int) + 2*l_M*T.d0 - l_M*T.dn + T.dt/2*l_M*alpha*T.dn)/(l_M + 0.5*alpha*l_M*T.dt) 
	d1[Local_Dirichlet] = 0


	# gather neighbor forces
	if size != 1:  # if serial, skip
		if num<=num_cri-1: # syn it
			F_int 	 = syn_cpus(size, rank, F_int ,len(Points), Local_nodes) # synchronize internal force
			# get the synchorinized value, the true one 
			d1  = (T.dt**2*(f_0 - F_int) + 2*l_M*T.d0 - l_M*T.dn + T.dt/2*l_M*alpha*T.dn)/(l_M + 0.5*alpha*l_M*T.dt) 
			d1[Local_Dirichlet] = 0
			Res_net_data(num,d1,shared_nodes,Local_nodes,rank, S=True, save = 'Dis')
		else:
			d1 = shared_update_displacement(d1,size,rank,[],Points,Local_nodes,F_int,T,\
								f_0,l_M,alpha, num, num_cri,dnn_every,shared_nodes)
			d1[Local_Dirichlet] = 0 # still give strong dirichlet bc, the shared nodes are hardly on the boundaries, even 
			# if they are, it still makes sense.
			Res_net_data(num,d1,shared_nodes,Local_nodes,rank, S=True, save = 'Dis')
			comm.Barrier() # wait until all update is done

	return d1
	

def shared_update_displacement(d1,size,rank,syn_rank_list,Points,Local_nodes,F_int,T,\
								f_0,l_M,alpha, num, num_cri,dnn_every,shared_nodes):
	if rank in syn_rank_list: # processors still to be synchronized		
		F_int = syn_cpus(size, rank, F_int ,len(Points), Local_nodes)
		return (T.dt**2*(f_0 - F_int) + 2*l_M*T.d0 - l_M*T.dn + T.dt/2*l_M*alpha*T.dn)/(l_M + 0.5*alpha*l_M*T.dt) 
	
	else:# other processors use the DNN prediction 
		if (num - (num_cri)) % (dnn_every) == 0: # alternating dnn step
			if rank ==  1:
				print('step num is: ' + str(num+1)+ ', current step is being predicted')
			return Load_DNN_data_dis(d1, rank, Local_nodes,shared_nodes,num)
		else: # synchronization correction steps
			F_int = syn_cpus(size, rank, F_int ,len(Points), Local_nodes)
			return (T.dt**2*(f_0 - F_int) + 2*l_M*T.d0 - l_M*T.dn + T.dt/2*l_M*alpha*T.dn)/(l_M + 0.5*alpha*l_M*T.dt) 
	


# Initial condition samping for the displacement
def IC_sampling(Points, min_dz, mag):
	d0 = np.zeros((len(Points)*3,1)) # initialize d0

	max_def = mag*min_dz # maximum deflection of the random IC profile

	s = np.random.uniform(max_def,0)
	dis_z  =  lambda x : s/25/25*x**2  # proposed initial dis function
	
	d0[0::3,0]  = 0 
	d0[1::3,0]  = 0 
	d0[2::3,0]  = dis_z(Points[:,0])

	return d0, dis_z