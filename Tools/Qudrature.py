import numpy as np

# n: order of accuracy needed
# ref:  Finite Element Method - Its Basis and Fundamentals (6th Edition), Zienkiewicz, O.C.; Taylor, R.L.; Zhu, J.Z.
# ref:  FIAT: https://github.com/FEniCS/fiat/blob/master/FIAT/quadrature_schemes.py
def Gauss_Legendre(n):
	if n == 2: # O(h^2)
		nodes   = np.array([[0.5854101966249685,0.1381966011250105,0.1381966011250105],
							[0.1381966011250105,0.5854101966249685,0.1381966011250105],
							[0.1381966011250105,0.1381966011250105,0.5854101966249685],
							[0.1381966011250105,0.1381966011250105,0.1381966011250105]])
		weights = np.array(	[0.25/6,0.25/6,0.25/6,0.25/6])
	if n == 3: # O(h^3)
		nodes   = np.array([[1./4,1./4,1./4],
							[1./2,1./6,1./6],
							[1./6,1./2,1./6],
							[1./6,1./6,1./2],
							[1./6,1./6,1./6]])
		weights = np.array(	[-4./5/6, 9./20/6, 9./20/6, 9./20/6, 9./20/6])

	if n == 4: #0(h^4)

		nodes = np.array([[0.0000000000000000, 0.5000000000000000, 0.5000000000000000],
					[0.5000000000000000, 0.0000000000000000, 0.5000000000000000],
					[0.5000000000000000, 0.5000000000000000, 0.0000000000000000],
					[0.5000000000000000, 0.0000000000000000, 0.0000000000000000],
					[0.0000000000000000, 0.5000000000000000, 0.0000000000000000],
					[0.0000000000000000, 0.0000000000000000, 0.5000000000000000],
					[0.6984197043243866, 0.1005267652252045, 0.1005267652252045],
					[0.1005267652252045, 0.1005267652252045, 0.1005267652252045],
					[0.1005267652252045, 0.1005267652252045, 0.6984197043243866],
					[0.1005267652252045, 0.6984197043243866, 0.1005267652252045],
					[0.0568813795204234, 0.3143728734931922, 0.3143728734931922],
					[0.3143728734931922, 0.3143728734931922, 0.3143728734931922],
					[0.3143728734931922, 0.3143728734931922, 0.0568813795204234],
					[0.3143728734931922, 0.0568813795204234, 0.3143728734931922]])

		w = np.arange(14, dtype=np.float64)
		w[0:6] = 0.0190476190476190
		w[6:10] = 0.0885898247429807
		w[10:14] = 0.1328387466855907
		w = w/6.0
		weights = w

	return nodes,weights