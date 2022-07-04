# Synchronization-avoiding-algorithms

Code references for the numerical examples in the paper:

> *Data-driven synchronization-avoiding algorithms in the
explicit distributed structural analysis of soft tissue, Guoxiang Grayson Tong and Daniele E. Schiavazzi, 2022, submitted*

## Code descriptions:

#### Mesh_info: 
Folder containing ```gmsh``` source file and unstructured cantilever meshes of a few levels of refinement.

#### Tools:
Code for building 3D elastodynamics solver using linear finite element, construction of LSTM-encoder-decoder network, etc.

#### ```Data_prepare.py```
Main script to run the explicit finite elemnent elastodynamics solver.

#### ```Shared_extraction.py```
Extract displacement degrees of freedom on all shared mesh nodes, from pre-computed numerical solutions.

#### ```Training.py```
Main training script for the LSTM-encoder-decoder network

#### ```Onliner_predictor.py```
Main script to perform the proposed synchronization-avoiding algorithm, using the pre-trained network model.

## Dependencies: 
- MPI for distributed computation: [```mpi4py```](https://pypi.org/project/mpi4py/)
- Meshing: [```gmsh```](https://gmsh.info/)
- Mesh processing [```meshio```](https://github.com/nschloe/meshio)
- Mesh partitioning: [```mgmetis```](https://github.com/chiao45/mgmetis)
- Neural network building: [```PyTorch```](https://pytorch.org/get-started/locally/)
