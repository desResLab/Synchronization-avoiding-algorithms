import numpy as np
from Tools.Mat_construction import *
from Tools.commons import *
from Tools.Distributed_tools import *
from Tools.DNN_prediction import *
import time


# solve dynamics by parallel solver, with cost measurement
def parallel_explicit_solver_dis(p, Cells, Points, Local_nodes, Local_Dirichlet,\
									T, Elas, l_M, alpha, size, rank, MODEL=False):
	n_basis = 4
	F_int 	 = np.zeros((len(Local_nodes)*3,1))  # dof-wise internal force allocation
	F_ext    = np.zeros((len(Local_nodes)*3,1))  # dof-wise external force allocation

	single_step_mat_vec = 0
	single_step_syn     = 0
	single_step_form    = 0

	# loop over all elemenet
	for i in range(len(Cells)):
		
		t_form = time.time()

		ele = Cells[i] # locate local nodes
		point = []
		for n_b in range(n_basis):
			point.append(Points[ele[n_b]])

		P     = np.array(point)
		_, Ke, 	F_e 	  = Local_MKF(p, n_basis, P, Elas, None, None, T.tn) 
		F_e = (F_e).reshape((len(F_e),1))

		# find the corresponding slices of the computed displacement vector
		ele_local    = local_mat_node(ele,Local_nodes)   
		ele_dof      = node_to_dof(3,[0,1,2],ele_local)  
		single_step_form += time.time() - t_form
		
		t_mv = time.time()
		F_int[ele_dof] += Ke @ (T.d0)[ele_dof]
		single_step_mat_vec += time.time() - t_mv

		F_ext[ele_dof] += F_e

	

	t_mv2 = time.time()
	# compute the unsynchronized value, to be used for dnn
	d1  = (T.dt**2*(F_ext - F_int) + 2*l_M*T.d0 - l_M*T.dn + T.dt/2*l_M*alpha*T.dn)/(l_M + 0.5*alpha*l_M*T.dt) 
	single_step_mat_vec += time.time() - t_mv2

	# strongly enforce the dirichlet bc
	d1[Local_Dirichlet] = 0

	# if false, use traditional synchoronization, otherwise, skip this part
	if MODEL== False:
		# gather neighbor forces
		if size != 1:  # if serial, skip
			
			t_syn = time.time()
			F_int 	 = syn_cpus(size, rank, F_int ,len(Points), Local_nodes) # synchronize internal force
			F_ext 	 = syn_cpus(size, rank, F_ext ,len(Points), Local_nodes) # synchronize external force
			single_step_syn += time.time() - t_syn

			t_mv3 = time.time()
			# get the synchorinized value, the true one 
			d1  = (T.dt**2*(F_ext - F_int) + 2*l_M*T.d0 - l_M*T.dn + T.dt/2*l_M*alpha*T.dn)/(l_M + 0.5*alpha*l_M*T.dt) 
			single_step_mat_vec += time.time() - t_mv3
			d1[Local_Dirichlet] = 0
			
	return d1, single_step_form, single_step_syn, single_step_mat_vec

