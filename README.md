# Synchronization-avoiding-algorithms

Code references for the numerical examples in the paper:

> [*Data-driven synchronization-avoiding algorithms in the
explicit distributed structural analysis of soft tissue, Guoxiang Grayson Tong and Daniele E. Schiavazzi, 2022, Computational Mechanics*](https://link-springer-com.proxy.library.nd.edu/article/10.1007/s00466-022-02248-w)

## Descriptions:

#### Mesh_info: 
Folder containing ```gmsh``` source file and a unstructured cantilever mesh 

#### Results: 
Folder will contain computed FEA solution, modelled solution, and mesh partitioning information

#### Tools:
Code for building 3D elastodynamics solver using linear finite element, construction of LSTM-encoder-decoder network, etc.

#### ```Data_prepare.py```
Main script to run the explicit finite element elastodynamics solver to gather training data.

#### ```Shared_extraction.py```
Extract displacement degrees of freedom on all shared nodes, from pre-computed numerical solutions.

#### ```Model_training.py```
Main training script for the LSTM-encoder-decoder network

#### ```Onliner_predictor.py```
Main script to perform the proposed synchronization-avoiding algorithm, using the pre-trained network model.

## Working flow:

A minimal working example is included in this repository. 
- Execute ```mpirun -np 2 python3 Data_prepare.py```. The processor-wise solutions will be saved to ```Results/Dynamics/```.
- Execute ```mpirun -np 2 python3 Shared_extraction.py```. The processor-wise shared degrees of freedom will be saved to ```Results/sol_on_shared/```.
- Execute ```mpirun -np 2 python3 Model_training.py```. The trained model, training/validation error, etc will be saved to ```Distributed_save/```.
- Execute ```mpirun -np 2 python3 Online_predictor.py```. The predicted solution will be saved to ```Results/Dynamics/```.
- Execute ```python3 plotter.py``` inside ```Results/```. The comparison picture can be seen as ```Comparison.pdf```.

Note: The hyperparameter setting in ```Model_training.py``` is primarily for fast training, so not optimal.

## Dependencies: 
- MPI for distributed computation: [```mpi4py```](https://pypi.org/project/mpi4py/)
- Meshing: [```gmsh```](https://gmsh.info/)
- Mesh processing [```meshio```](https://github.com/nschloe/meshio)
- Mesh partitioning: [```mgmetis```](https://github.com/chiao45/mgmetis)
- Neural network building: [```PyTorch```](https://pytorch.org/get-started/locally/)
- Code reference for LSTM-encoder-decoder network: [```LSTM```](https://github.com/lkulowski/LSTM_encoder_decoder)
