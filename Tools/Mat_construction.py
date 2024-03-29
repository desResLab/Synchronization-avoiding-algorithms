from Tools.Shape_function_Deriv import *
from Tools.Shape_function_Deriv import *
from Tools.Qudrature import *
from Tools.commons import *
from Tools.Distributed_tools import *
import numpy as np
from scipy.sparse import csr_matrix

#_____________________________________________________________________
# this function generates the local elementwise stiffness matrix, force vector
# input: 
#		 n_basis: number of basis per element
#		 P: 4 node coordinates of the selected element, 4x3 array
#        Elas: collection of problem dependeent parameters
#		 facet: Neumann facets 	
#        Neumann: traction free for the current problem
#        t      : current time
# outputs:  
#        M: if true, local mass matrix 
# 		 K: local stiffness matrix
#		 F: local force vector

def Local_MKF(deg, n_basis,P, Elas, facet, Neumann, t):
	
	n_eq  = int(3*n_basis) # number of equations
	M,K,F =   np.zeros((n_eq,n_eq)),np.zeros((n_eq,n_eq)),np.zeros((n_eq,1)) 
	
	if deg == 1:
		n_quad = 2 # However, one-point rule is enough
	if deg == 2:
		n_quad = 2 # dynamics problem for p=2 TBD

	# Gaussian nodes and weights
	xi_quad, w_quad = Gauss_Legendre(n_quad) 

	for xi in range(len(xi_quad[:,0])): # Gaussian node loop
		Shape_fun = Shape_Function(deg, xi_quad[xi,:])
		Jac   = Jacobian(deg, P,xi_quad[xi,:]) 
		detJ  = np.linalg.det(Jac)
		invJ  = np.linalg.inv(Jac)

		N_xyz = Shape_Deri(deg, xi_quad[xi,:]) @ invJ         # physical derivatives of the shape functions n_basisx3
		X = IsoparametricMap(deg,P,xi_quad[xi,:])             # physical coodinate
		
		f_loc = Elas.f(X,t)                                   # rhs vector

		for i in range(0,n_basis): # local basis loop
			Bi = np.array([[N_xyz[i,0], 0.0, 0.0],
						   [0.0, N_xyz[i,1], 0.0],
						   [0.0, 0.0, N_xyz[i,2]],
						   [0.0, N_xyz[i,2], N_xyz[i,1]],
						   [N_xyz[i,2], 0.0, N_xyz[i,0]],
						   [N_xyz[i,1],N_xyz[i,0], 0.0]] )
			for j in range(0,n_basis):
				Bj = np.array([[N_xyz[j,0], 0.0, 0.0],
						      [0.0, N_xyz[j,1], 0.0],
						      [0.0, 0.0, N_xyz[j,2]],
						      [0.0, N_xyz[j,2], N_xyz[j,1]],
						      [N_xyz[j,2], 0.0, N_xyz[j,0]],
						      [N_xyz[j,1],N_xyz[j,0], 0.0]] )
				k_loc = np.transpose(Bi) @ Elas.D() @ Bj * detJ * w_quad[xi]
				m_loc = Shape_fun[i] * Elas.rho * Shape_fun[j] * detJ * w_quad[xi]
				
				for A in [0,1,2]:
					for B in [0,1,2]:
						p = 3*i + A
						q = 3*j + B
						M[p,q] += m_loc*dirac(A,B)
						K[p,q] += k_loc[A,B]

			for C in [0,1,2]:
				p = 3*i + C
				F[p] += Shape_fun[i] * f_loc[C] * detJ * w_quad[xi] 
	
	# Neuamnn BC tbd
	return M,K,F.flatten()

# local stiffness matrix formations
def Local_K_coronary(deg, n_basis,P, Elas):
	
	n_eq =int(3*n_basis) # number of equations
	K    =   np.zeros((n_eq,n_eq))
	
	if deg == 1:
		n_quad = 2
	if deg == 2:
		n_quad = 2   

	xi_quad, w_quad = Gauss_Legendre(n_quad)
	for xi in range(len(xi_quad[:,0])):
		Shape_fun = Shape_Function(deg, xi_quad[xi,:])
		Jac   = Jacobian(deg, P,xi_quad[xi,:]) 
		detJ  = np.linalg.det(Jac)
		invJ  = np.linalg.inv(Jac)

		N_xyz = Shape_Deri(deg, xi_quad[xi,:]) @ invJ         # physical derivatives of the shape functions n_basisx3
		
		for i in range(0,n_basis):
			Bi = np.array([[N_xyz[i,0], 0.0, 0.0],
						   [0.0, N_xyz[i,1], 0.0],
						   [0.0, 0.0, N_xyz[i,2]],
						   [0.0, N_xyz[i,2], N_xyz[i,1]],
						   [N_xyz[i,2], 0.0, N_xyz[i,0]],
						   [N_xyz[i,1],N_xyz[i,0], 0.0]] )
			for j in range(0,n_basis):
				Bj = np.array([[N_xyz[j,0], 0.0, 0.0],
						      [0.0, N_xyz[j,1], 0.0],
						      [0.0, 0.0, N_xyz[j,2]],
						      [0.0, N_xyz[j,2], N_xyz[j,1]],
						      [N_xyz[j,2], 0.0, N_xyz[j,0]],
						      [N_xyz[j,1],N_xyz[j,0], 0.0]] )
				k_loc = np.transpose(Bi) @ Elas.D() @ Bj * detJ * w_quad[xi]				
				for A in [0,1,2]:
					for B in [0,1,2]:
						p = 3*i + A
						q = 3*j + B
						K[p,q] += k_loc[A,B]

	return K

# local stiffness matrix assembly
def Local_assembly_for_stiffness(local_node_list, Cell, Points, deg, n_basis, elas, rank):
	l = len(local_node_list)
	K =  np.zeros((l*3, l*3))
	for i in range(len(Cell)):
		if rank==0:
			print('i=' +  str(i+1) + '/' + str(len(Cell)))
		ele = Cell[i] # locate local nodes
		point     = []
		global_ID = []
		for n_b in range(n_basis):
			point.append(Points[ele[n_b]])
			global_ID.append(ele[n_b])

		Pt               = np.array(point)
		gID              = np.array(global_ID)
		Local_Ke         = Local_K_coronary(deg, n_basis, Pt, elas) 

		Local_ID = local_mat_node(gID, local_node_list)
		for A in [0,1,2]:              # loop over dimension x,y,z
			for a in range(n_basis):   # loop over local basis
				p = 3*a + A            # local matrix id
				P = (node_to_dof(3, [A] , [Local_ID[a]]))[0] # global equation number, global dof
				for B in [0,1,2]:
					for b in range(n_basis):
						q = 3*b + B
						Q = (node_to_dof(3, [B] , [Local_ID[b]]))[0] # global equation number, global dof
						K[P,Q] += Local_Ke[p,q]

	return csr_matrix(K)



# global_assembly, only used for steady solver and modal analysis
def Global_Assembly(deg, Cells, Points, Dirichlet, elas, t, Facets=None, Neumann=None, steady=False):
	
	if deg == 1:
		n_basis = 4
	elif deg == 2:
		n_basis = 10

	# global initialization
	M,K,F =  np.zeros((len(Points)*3, len(Points)*3)), np.zeros((len(Points)*3, len(Points)*3)), np.zeros((len(Points)*3,1))

	for ele in Cells:
		point = []
		global_ID = []
		for n_b in range(n_basis):
			point.append(Points[ele[n_b]])
			global_ID.append(ele[n_b])

		point     = np.array(point)
		global_ID = np.array(global_ID)

		Me, Ke, Fe = Local_MKF(deg,n_basis, point, elas, None, Neumann, t)

		for A in [0,1,2]:              # loop over dimension x,y,z
			for a in range(n_basis):   # loop over local basis
				p = 3*a + A            # local matrix id
				P = (node_to_dof(3, [A] , [global_ID[a]]))[0] # global equation number, global dof
				if P not in Dirichlet:
					for B in [0,1,2]:
						for b in range(n_basis):
							q = 3*b + B
							Q = (node_to_dof(3, [B] , [global_ID[b]]))[0] # global equation number, global dof
							if Q not in Dirichlet:
								M[P,Q] += Me[p,q]
								K[P,Q] += Ke[p,q]
							else:
								if steady == False:
									F[P] -= (Me[p,q]* 0 +Ke[p,q]* 0)  # loading is t-independent, not general, tbd
								else:
									F[P] -= (Ke[p,q]* 0)  # loading is t-independent, not general, tbd

					F[P] += Fe[p]
	return M,K,F

# Global_assembly without boundary conditions given, just for initializations, mass lumping and pre-assembly
def Global_Assembly_no_bc(deg, Cells, Points, elas, t):
	
	if deg == 1:
		n_basis = 4
	elif deg == 2:
		n_basis = 10

	# global initialization
	M,K,F =  np.zeros((len(Points)*3, len(Points)*3)), np.zeros((len(Points)*3, len(Points)*3)), np.zeros((len(Points)*3,1))

	for ele in Cells:
		point = []
		global_ID = []
		for n_b in range(n_basis):
			point.append(Points[ele[n_b]])
			global_ID.append(ele[n_b])

		point     = np.array(point)
		global_ID = np.array(global_ID)
		Me, Ke, Fe = Local_MKF(deg,n_basis, point, elas, None, None, t)

		for A in [0,1,2]:              # loop over dimension x,y,z
			for a in range(n_basis):   # loop over local basis
				p = 3*a + A            # local matrix id
				P = (node_to_dof(3, [A] , [global_ID[a]]))[0] # global equation number, global dof
				F[P] += Fe[p]
				for B in [0,1,2]:
					for b in range(n_basis):
						q = 3*b + B
						Q = (node_to_dof(3, [B] , [global_ID[b]]))[0] # global equation number, global dof
						M[P,Q] += Me[p,q]
						K[P,Q] += Ke[p,q]
	return M,K,F