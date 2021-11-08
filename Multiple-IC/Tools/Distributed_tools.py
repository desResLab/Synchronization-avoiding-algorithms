# this code contains tools for the distributed solver
import numpy as np
from Tools.commons import *
from Tools.Mat_construction import *
from Tools.Qudrature import *
#from Tools.DNN_predication import *

import meshio
from mpi4py import MPI
import time
from numpy import genfromtxt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# localize global nodal and element number of a particular processor
def rankwise_dist(rank,recvbuf, Points, Cells):
	ele_list   = []
	nodal_list = []

	for idx, node in enumerate(recvbuf):
		if node == rank:
			ele_list.append(idx)
			for n in Cells[idx]:
				if n not in nodal_list:
					nodal_list.append(n)
	return ele_list, nodal_list	


# This function finds the shared nodes for each processor, note that one processor can share
# data from each of the rest processors
def find_shared_nodes(rank,size,rank_nodal_num,rank_nodal_list):
	shared_nodes = []
	my_nodes     = rank_nodal_list[rank]
	my_num       = rank_nodal_num[rank]
	# print('My rank is:'+str(rank)+', number of nodes is:'+str(my_num)+', node list is:'+str(my_nodes))

	for r in range(size):
		if r != rank:
			for idx in rank_nodal_list[r]:
				if idx in my_nodes and idx not in shared_nodes:
					shared_nodes.append(idx)
	return shared_nodes 


# find the global sorted shared nodes
def sort_shared(G_shared_nodes):
	sorted = []
	for i in G_shared_nodes:
		for j in i:
			if j not in sorted:
				sorted.append(j)

	return np.sort(sorted)


#function to find local dirichlet array(dof-wise)
def Dirichlet_rank_dist(D_node, Local_N_list):
	local_DOF = []

	for idx,node in enumerate(Local_N_list):
		if node in D_node:
			local_DOF.append(idx)

	return node_to_dof(3, [0,1,2], local_DOF)


# function to map global nodal number to rankwised nodal number
def local_mat_node(G_ID, L_N):
	rank_mat_ID = []
	for g in G_ID:
		for idx, node in enumerate(L_N):
			if node==g:
				rank_mat_ID.append(idx)
				continue
	return rank_mat_ID


# this function gather forces in shared nodes 
def syn_cpus(size, rank, f, L_g, Local_nodes):
	# gather local contributions
	rank_force           = comm.gather(f,root=0)           # gather all internal forces
	rank_local_node_list = comm.gather(Local_nodes,root=0) # gather all local nodal number
	
	# synchronization process in the root node
	if rank == 0:
		f_global = np.zeros((L_g*3,1))  # dof-wise internal force allocationn
		for i in range(size):
			f_global[node_to_dof(3,[0,1,2],rank_local_node_list[i])]  += rank_force[i] 
	else:
		f_global = None # buffer
	
	# send it back
	f_global = comm.bcast(f_global, root=0)
	return f_global[node_to_dof(3,[0,1,2],Local_nodes)]



# update using dnn model
def shared_update(num, num_cri,  syn_rank_list, size, rank, a1, F_int, F_ext, Points, Local_nodes, shared_nodes
					,alpha, mass,T, v_half, dnn_every,M_flag):
	if rank in syn_rank_list: 
		
		F_int = syn_cpus(size, rank, F_int ,len(Points), Local_nodes)
		F_ext = syn_cpus(size, rank, F_ext ,len(Points), Local_nodes)
		
		return (F_ext - F_int - alpha*mass*v_half) / (mass + (T.tn_plus_1() - T.tn_plus_half())*alpha*mass)
	
	else:  # other processors use the DNN prediction 
		if (num - (num_cri+1)) % (dnn_every + 1) == 0: # alternating dnn step
			if M_flag == 'acc':
				return Load_DNN_data_acc(a1, size,rank,Local_nodes,shared_nodes,num)
			else:
				return Load_DNN_data_acc_inc(a1, size,rank,Local_nodes,shared_nodes,num)
		else:
			F_int = syn_cpus(size, rank, F_int ,len(Points), Local_nodes)
			F_ext = syn_cpus(size, rank, F_ext ,len(Points), Local_nodes)
			
			return (F_ext - F_int - alpha*mass*v_half) / (mass + (T.tn_plus_1() - T.tn_plus_half())*alpha*mass) 
			
		

# function to read learned data from file and update the value on shared nodes
def Load_DNN_data_acc(a1, size,rank,Local_nodes,shared_nodes,step):
	if rank == 0:
		print('Current step is being predicted!, step number is: ' + str(step+1) )
	learned_acc = Res_predicting(rank,size,step)
	local_shared_dof = node_to_dof(3, [0,1,2], local_mat_node(shared_nodes,Local_nodes))
	a1[local_shared_dof] = learned_acc
	return a1


# function to read learned data from file and update the value on shared nodes, for acc-increment
def Load_DNN_data_acc_inc(a1, size,rank,Local_nodes,shared_nodes,step):
	if rank == 0:
		print('Current step is being predicted(by inc)!, step number is: ' + str(step+1) )
	learned_acc = Res_predicting_inc(rank,size,step)
	local_shared_dof = node_to_dof(3, [0,1,2], local_mat_node(shared_nodes,Local_nodes))
	a1[local_shared_dof] = learned_acc
	return a1



# # parallel solver using the method from dyna, based on displacement
# def parallel_explicit_solver_dis(deg, n_basis, num, num_cri, size, rank, Cells,Points,
# 									Local_nodes,shared_nodes,Local_Dirichlet,T,elas,mass, \
# 										alpha,dnn_every,Facet,Neumann,M_flag='dis'):
	
# 	F_int 	 = np.zeros((len(Local_nodes)*3,1))  # dof-wise internal force allocation
# 	F_ext    = np.zeros((len(Local_nodes)*3,1))  # dof-wise external force allocation

# 	# half-step velocity
# 	v_first_half = (T.d0 - T.dn1)/T.dt

# 	for i in range(len(Cells)):
# 		ele       = Cells[i]

# 		point = []
# 		for n_b in range(n_basis):
# 			point.append(Points[ele[n_b]])
# 		P     = np.array(point)
# 		_, Ke,f_ext 	  = Local_MKF(deg, n_basis, P, elas, Facet, Neumann, T.tn) # f exterior is fixed in this problem

# 		# find the corresponding slices of the computed displacement vector
# 		ele_local    = local_mat_node(ele,Local_nodes)   
# 		ele_dof      = node_to_dof(3,[0,1,2],ele_local)  
		
# 		# strongly impose Dirichlet BC
# 		for j in ele_dof:
# 			if j in Local_Dirichlet:
# 				T.d0[j]  = 0
# 		# update global internal force vector
# 		F_int[ele_dof] += Ke @ T.d0[ele_dof]
# 		F_ext[ele_dof,0] += f_ext

# 	a1 =  (F_ext - F_int - alpha*mass*v_first_half) / mass
# 	a1[Local_Dirichlet] = 0
# 	v1 = v_first_half + T.dt*a1
# 	d1 = T.d0 + T.dt*v1

# 	# syn
# 	if size != 1: # if serial, skip
# 		if num <= num_cri: #  syn it
# 			F_int 	 = syn_cpus(size, rank, F_int ,len(Points), Local_nodes) # synchronize internal force
# 			F_ext    = syn_cpus(size, rank, F_ext, len(Points), Local_nodes) # synchronize external force

# 			a1 =  (F_ext - F_int - alpha*mass*v_first_half) / mass
# 			a1[Local_Dirichlet] = 0
# 			v1 = v_first_half + T.dt*a1
# 			d1 = T.d0 + T.dt*v1

# 			# save syn-ed internal force for resnet learning
# 			#Res_net_data(num,a1,shared_nodes,Local_nodes,rank, S=True, save = 'a')
		
# 		else: # After the crirical step, start dnn predication
# 			print('error!')
# 			# a1    =  shared_update(num, num_cri, [], size, rank, a1, F_int, F_ext, Points, Local_nodes,shared_nodes, \
# 			# 										alpha, mass, T, v_half, dnn_every, M_flag='acc')  # update after having robust dnn model
			
# 			# # strongly impose Dirichlet BC to acc
# 			# a1[Local_Dirichlet] = 0
# 			# v1 =  v_half + (T.tn_plus_1() - T.tn_plus_half()) * a1
# 			# Res_net_data(num,a1,shared_nodes,Local_nodes,rank, S=True, save = 'a')
# 			# comm.Barrier() # wait until all update is done


# 	return d1,v1,a1





# parallel solver using the method from the GPU papaer, based on acclearation
def parallel_explicit_solver_acc(deg, n_basis, num, num_cri, size, rank, Cells,Points,
									Local_nodes,shared_nodes,Local_Dirichlet,T,elas,mass, \
										alpha,dnn_every,Facet,Neumann,M_flag='acc'):

	F_int 	 = np.zeros((len(Local_nodes)*3,1))  # dof-wise internal force allocation
	F_ext    = np.zeros((len(Local_nodes)*3,1))  # dof-wise external force allocation
	# update new displacement vector: step3 (2.5, 2.6) of the pp 
	v_half = T.v0 + (T.tn_plus_half() - T.tn) * T.a0
	d1     = T.d0 + T.dt*v_half

	# Step 4 of the paper,  find global contribution of internal and external forces
	for i in range(len(Cells)):
		ele       = Cells[i]

		point = []
		for n_b in range(n_basis):
			point.append(Points[ele[n_b]])
		P     = np.array(point)
		_, Ke,f_ext 	  = Local_MKF(deg, n_basis, P, elas, Facet, Neumann, T.tn) # f exterior is fixed in this problem

		# find the corresponding slices of the computed displacement vector
		ele_local    = local_mat_node(ele,Local_nodes)   
		ele_dof      = node_to_dof(3,[0,1,2],ele_local)  
		
		# strongly impose Dirichlet BC
		for j in ele_dof:
			if j in Local_Dirichlet:
				d1[j]  = 0
		# update global internal force vector
		F_int[ele_dof] += Ke @ d1[ele_dof]
		F_ext[ele_dof,0] += f_ext

	a1 =  (F_ext - F_int - alpha*mass*v_half) / (mass + (T.tn_plus_1() - T.tn_plus_half())*alpha*mass)
	
	# strongly impose Dirichlet BC to acc
	a1[Local_Dirichlet] = 0
	v1 =  v_half + (T.tn_plus_1() - T.tn_plus_half()) * a1

	Res_net_data(num,a1,shared_nodes,Local_nodes,rank, S=False, save = 'a')


	# gather neighor forces
	if size != 1: # if serial, skip
		if num <= num_cri: #  syn it
			F_int 	 = syn_cpus(size, rank, F_int ,len(Points), Local_nodes) # synchronize internal force
			F_ext    = syn_cpus(size, rank, F_ext, len(Points), Local_nodes) # synchronize external force

			a1 =  (F_ext - F_int - alpha*mass*v_half) / (mass + (T.tn_plus_1() - T.tn_plus_half())*alpha*mass)
			# strongly impose Dirichlet BC to acc
			a1[Local_Dirichlet] = 0
			v1 =  v_half + (T.tn_plus_1() - T.tn_plus_half()) * a1

			# save syn-ed internal force for resnet learning
			Res_net_data(num,a1,shared_nodes,Local_nodes,rank, S=True, save = 'a')
		
		else: # After the crirical step, start dnn predication
			a1    =  shared_update(num, num_cri, [], size, rank, a1, F_int, F_ext, Points, Local_nodes,shared_nodes, \
													alpha, mass, T, v_half, dnn_every, M_flag)  # update after having robust dnn model
			
			# strongly impose Dirichlet BC to acc
			a1[Local_Dirichlet] = 0
			v1 =  v_half + (T.tn_plus_1() - T.tn_plus_half()) * a1
			Res_net_data(num,a1,shared_nodes,Local_nodes,rank, S=True, save = 'a')
			comm.Barrier() # wait until all update is done

	# note that after num_cri, the saved data is no longer exact
	return d1,v1,a1


# function to store the shared nodes data for ddn learning
# num: current time step
# a: rankwise solution
# shared_nodes: global shared nodal list
# Local_N_list: rankwise global nodal list
# S: logical var: True: output syncronized data, False: output unsyncronized data
# save: which file to save, default is acc
def Res_net_data(num,a,shared_nodes,Local_N_list,rank, S=True, save='Acc'):

	shared_dof = a[node_to_dof(3,[0,1,2],local_mat_node(shared_nodes,Local_N_list))]
	if S == True:
		save_name = 'Results/Res_Net_data/S/Step_'+str(num)+'_Rank='+str(rank)+'_'+save+'.csv' # save syn-ed data
		np.savetxt(save_name,shared_dof,delimiter=',')
	else:
		save_name = 'Results/Res_Net_data/US/Step_'+str(num)+'_Rank='+str(rank)+'_'+save+'.csv' # save un-syn-ed data
		np.savetxt(save_name,shared_dof,delimiter=',')

	return 0



# function to write results to rankwise file
def Write_to_vtk(n_basis, d,v,a,Points,Local_Cells,Local_N_list,rank,tn,num):
	rank_wise_cell   = np.zeros((len(Local_Cells),n_basis))
	rank_wise_point  = np.zeros((len(Local_N_list),3))

	# localize points and cells
	for idx,ele in enumerate(Local_Cells):
		local_nodes = local_mat_node(ele,Local_N_list)
		rank_wise_cell[idx,:]=local_nodes
		for i in ele:
			rank_wise_point[local_mat_node([i],Local_N_list),:] = Points[i,:]

	if n_basis == 4:
		rank_wise_cell = [("tetra",rank_wise_cell)]
	elif n_basis == 10:
		rank_wise_cell = [("tetra10",rank_wise_cell)]
	save_name = 'Results/Rankwise-data/R/rank-'+str(rank)+'-num='+str(num)+ '.vtk' 

	if v!=None and a!=None:
		meshio.write_points_cells(save_name, rank_wise_point , rank_wise_cell, {'d-x':d[0::3],\
																				'd-y':d[1::3],\
																				'd-z':d[2::3],\
																				'v-x':v[0::3],\
																				'v-y':v[1::3],\
																				'v-z':v[2::3],\
																				'a-x':a[0::3],\
																				'a-y':a[1::3],\
																				'a-z':a[2::3]})
	if v == None and a == None:
		meshio.write_points_cells(save_name, rank_wise_point , rank_wise_cell, {'d-x':d[0::3],\
																				'd-y':d[1::3],\
																				'd-z':d[2::3]})


	return 0

# write all results in a single file
def Write_to_vtk_single(size, Cells, Points,save_int, test_num, flag = 9):

	def find_index(p,P):
		for i in range(len(P)):
			if np.isclose(p[0],P[i,0]) and np.isclose(p[1],P[i,1]) and np.isclose(p[2],P[i,2]):
				return i
				break

	if flag==9: # that is the default setting, for acclearation based solver

		global_data     = np.zeros((len(Points),9))  # dx,dy,dz,vx,vy,vz,ax,ay,az
		print("start to write results in a single vtk file")
		for num_step in range(0,test_num,save_int):
			for rank in range(size):
				file    = 'Results/Rankwise-data/R/rank-'+str(rank)+'-num='+str(num_step)+ '.vtk' 
				data    = meshio.read(file)

				xyz        = data.points
				point_data = data.point_data

				for itr, point in enumerate(xyz):

					global_row = find_index(point, Points)
					global_data[global_row,0] = (point_data['d-x'])[itr]
					global_data[global_row,1] = (point_data['d-y'])[itr]
					global_data[global_row,2] = (point_data['d-z'])[itr]
					global_data[global_row,3] = (point_data['v-x'])[itr]
					global_data[global_row,4] = (point_data['v-y'])[itr]
					global_data[global_row,5] = (point_data['v-z'])[itr]
					global_data[global_row,6] = (point_data['a-x'])[itr]
					global_data[global_row,7] = (point_data['a-y'])[itr]
					global_data[global_row,8] = (point_data['a-z'])[itr]

			G_data = np.array(global_data)
			save_name = 'Results/Rankwise-data/S/Num='+str(num_step)+ '.vtk' 
			meshio.write_points_cells(save_name, Points , Cells, {  'd-x': G_data[:,0],
																	'd-y': G_data[:,1],\
																	'd-z': G_data[:,2],\
																	'v-x': G_data[:,3],\
																	'v-y': G_data[:,4],\
																	'v-z': G_data[:,5],\
																	'a-x': G_data[:,6],\
																	'a-y': G_data[:,7],\
																	'a-z': G_data[:,8]})		
	if flag == 3: # for the displacement solver
		global_data     = np.zeros((len(Points),3))  # dx,dy,dz
		print("start to write results in a single vtk file")
		for num_step in range(0,test_num,save_int):
			for rank in range(size):
				file    = 'Results/Rankwise-data/R/rank-'+str(rank)+'-num='+str(num_step)+ '.vtk' 
				data    = meshio.read(file)

				xyz        = data.points
				point_data = data.point_data

				for itr, point in enumerate(xyz):

					global_row = find_index(point, Points)
					global_data[global_row,0] = (point_data['d-x'])[itr]
					global_data[global_row,1] = (point_data['d-y'])[itr]
					global_data[global_row,2] = (point_data['d-z'])[itr]

			G_data = np.array(global_data)
			save_name = 'Results/Rankwise-data/S/Num='+str(num_step)+ '.vtk' 
			meshio.write_points_cells(save_name, Points , Cells, {  'd-x': G_data[:,0],
																	'd-y': G_data[:,1],\
																	'd-z': G_data[:,2]})		

	return 0
