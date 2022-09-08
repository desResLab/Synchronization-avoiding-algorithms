import numpy as np
from Tools.Mat_construction import *
from Tools.commons import *
from Tools.Distributed_tools import *
from Tools.DNN_prediction import *
import time

# Dynamics solver with pre-assembled K and F
def parallel_explicit_solver_dis_pre(LocalK, F_rankwise, Points, Local_nodes, Local_Dirichlet,\
									T, Elas, l_M, alpha, size, rank, MODEL=False):

	F_int  = LocalK.dot(T.d0) # internal force computation
	F_ext = (F_rankwise * linear_ramp(T.tn)).reshape((len(F_rankwise),1)) # ramp the external force 
	l_M = l_M.reshape((len(l_M),1)) # lumped mass vector

	# compute the unsynchronized value, to be used for dnn
	d1  = (T.dt**2*(F_ext - F_int) + 2*l_M*T.d0 - l_M*T.dn + T.dt/2*l_M*alpha*T.dn)/(l_M + 0.5*alpha*l_M*T.dt) 
	
	# strongly enforce the dirichlet bc
	d1[Local_Dirichlet] = 0

	if MODEL== False: # if no data-driven model is used, do synchronization. Otherwise, skip and use the model prediction
		
		# gather neighbor forces
		if size != 1:  # if serial, skip
			F_int 	 = syn_cpus(size, rank, F_int ,len(Points), Local_nodes) # synchronize internal force
			
			# get the synchorinized value, the true one 
			d1  = (T.dt**2*(F_ext - F_int) + 2*l_M*T.d0 - l_M*T.dn + T.dt/2*l_M*alpha*T.dn)/(l_M + 0.5*alpha*l_M*T.dt) 
			
			# strongly enforce the dirichlet bc again if needed
			d1[Local_Dirichlet] = 0
	
	return d1
	