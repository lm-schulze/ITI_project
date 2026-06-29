# Information Theory and Inference project: Maps of random walks on complex networks reveal community structure
Course project for the course *Information Theory and Inference* at the University of Padova, A.Y. 2025/2026. The project is based on the paper ["Maps of random walks on complex networks reveal community structure"](https://doi.org/10.1073/pnas.0706851105). Details to be added.
Contributors:
- Marco Foster ([@MarcoTFoster](https://github.com/MarcoTFoster))
- Laura Schulze ([@lm-schulze](https://github.com/lm-schulze))
- Savina Tsichli ([@savinats](https://github.com/savinats))

This project implements and compares two approaches to community detection based on the map equation: the classic **Infomap** algorithm, which uses a 
greedy stochastic search to optimize the map equation directly, and **Neuromap**, which learns a soft cluster assignment matrix end-to-end via a neural network and gradient descent. 

For Infomap, we implement the core map equation (description length computation), the search algorithm for finding optimal partitions, and the submodule refinement step, validating our implementation against igraph's `community_infomap` and the official Infomap package.

For Neuromap, we test three encoder architectures; a Graph Convolutional Network (GCN), GraphSAGE, and FastGCN, in order to compare how architecture choice affects the resulting communities and codelength.

The project consists of the following files and folders (to be updated as the project progresses):
- **test_graphs/**: folder containing test graphs exported as GraphML, (weighted/unweighted, directed/undirected); as well as visualizations of each graph as .pdf
- **src/**: folder containing the implemented functions for infomap
  - `map_equation.py`: function definitions shared between the notebooks
  - `optimize.py`: functions for computing & updating description length, and performing the search for an optimal community partition.
  - `utils.py`: helper functions used to generate & visualise test networks.
- **notebooks/**: folder containing the notebooks used to test and compare the different components of the infomap implementation, using igraph
  - `Test_networks.ipynb`: Jupyter notebook containing the SBM-based graph generation function used to generate the test graphs.
  - `Test_infomap.ipynb`: Jupyter notebook testing a self-implemented version of the Infomap description length computation via map equation, as well as custom update functions.
  - `SearchAlgorithm.ipynb`: Jupyter notebook for testing the functions needed for implementing the search algorithm to find the optimal community partition wrt. description length.
  - `SubmoduleRefinement.ipynb`: Jupyter notebook testing the recursive submodule refinement step and the full infomap workflow, validated against igraph's `community_infomap` and against the ground-truth partition.
  - `uniform.ipynb`: map equation with uniform recorded teleportation (matches Laura's setup, sanity check).
  - `non-uniform.ipynb`: map equation with smart unrecorded teleportation (tutorial's recommended scheme).
