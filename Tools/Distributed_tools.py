# this code contains tools for the distributed solver
import numpy as np
from Tools.commons import *
from Tools.Qudrature import *
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
		f_global = np.zeros((L_g*3,1))  # global Force allocationn
		for i in range(size):
			f_global[node_to_dof(3,[0,1,2],rank_local_node_list[i])]  += rank_force[i] 
	else:
		f_global = None # buffer
	
	# send it back
	f_global = comm.bcast(f_global, root=0)
	return f_global[node_to_dof(3,[0,1,2],Local_nodes)]


# function to write results to rankwise file
def Write_to_vtk(n_basis, d, Points,Local_Cells,Local_N_list,rank,num, save_path):
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
	save_name = save_path+'rank-'+str(rank)+'-num='+str(num)+ '.vtk' 

	meshio.write_points_cells(save_name, rank_wise_point , rank_wise_cell,     {'d-x':d[0::3],\
																				'd-y':d[1::3],\
																				'd-z':d[2::3]})

	return 0


# function to write results to rankwise file
def Write_to_vtk_surface(n_basis, d, Points,Local_Cells,Local_N_list,rank,num, save_path):
	rank_wise_cell   = np.zeros((len(Local_Cells),n_basis))
	rank_wise_point  = np.zeros((len(Local_N_list),3))

	# localize points and cells
	for idx,ele in enumerate(Local_Cells):
		local_nodes = local_mat_node(ele,Local_N_list)
		rank_wise_cell[idx,:]=local_nodes
		for i in ele:
			rank_wise_point[local_mat_node([i],Local_N_list),:] = Points[i,:]
	rank_wise_cell = [('triangle',rank_wise_cell)]
	save_name = save_path+'rank-'+str(rank)+'-num='+str(num)+ '.vtk' 

	meshio.write_points_cells(save_name, rank_wise_point , rank_wise_cell,     {'d-x':d[0::2],\
																				'd-y':d[1::2],\
																				})

	return 0


# write all results in a single file
def Write_to_vtk_single(size, Cells, Points,save_int, test_num, path):

	def find_index(p,P):
		for i in range(len(P)):
			if np.isclose(p[0],P[i,0]) and np.isclose(p[1],P[i,1]) and np.isclose(p[2],P[i,2]):
				return i
				break


	global_data     = np.zeros((len(Points),3))  # dx,dy,dz
	print("start to write results in a single vtk file")
	for num_step in range(0,test_num,save_int):
		for rank in range(size):
			file    = path+'rank-'+str(rank)+'-num='+str(num_step)+ '.vtk' 
			data    = meshio.read(file)

			xyz        = data.points
			point_data = data.point_data

			for itr, point in enumerate(xyz):

				global_row = find_index(point, Points)
				global_data[global_row,0] = (point_data['d-x'])[itr]
				global_data[global_row,1] = (point_data['d-y'])[itr]
				global_data[global_row,2] = (point_data['d-z'])[itr]

		G_data = np.array(global_data)
		save_name = path+'Single-Num='+str(num_step)+ '.vtk' 
		meshio.write_points_cells(save_name, Points , Cells, {  'd-x': G_data[:,0],
																'd-y': G_data[:,1],\
																'd-z': G_data[:,2]})		

	return 0

