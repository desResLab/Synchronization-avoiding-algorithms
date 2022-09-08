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
			f_global[node_to_dof(3,[0,1,2],rank_local_node_list[i])]  += rank_force[i] # add neighbor contributions
	else:
		f_global = None # buffer
	
	# send it back
	f_global = comm.bcast(f_global, root=0)
	return f_global[node_to_dof(3,[0,1,2],Local_nodes)] # localize rankwise information again


