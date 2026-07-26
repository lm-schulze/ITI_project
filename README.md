# Information Theory and Inference project: Maps of random walks on complex networks reveal community structure
Course project for the course *Information Theory and Inference* at the University of Padova, A.Y. 2025/2026.

Contributors:
- Marco Foster ([@MarcoTFoster](https://github.com/MarcoTFoster))
- Laura Schulze ([@lm-schulze](https://github.com/lm-schulze))
- Savina Tsichli ([@savinats](https://github.com/savinats))

This project implements and compares two approaches to community detection based on the map equation: the classic **Infomap** algorithm (see ["Maps of random walks on complex networks reveal community structure"](https://doi.org/10.1073/pnas.0706851105).), which uses a greedy stochastic search to optimize the map equation directly, and **Neuromap** (see ["The Map Equation Goes Neural: Mapping Network Flows with Graph Neural Networks"](https://doi.org/10.52202/079017-0554)), which learns a soft cluster assignment matrix end-to-end via a graph neural network and gradient descent. 

For Infomap, we implement the core map equation (description length computation), the search algorithm for finding optimal partitions, and the submodule refinement step, validating our implementation against python igraph and the Infomap package.

For Neuromap, we test four GNN architectures; Graph Convolutional Network (GCN), Graph Isomorphism Network (GIN),  Graph Attention Network (GAT), and GraphSAGE, in order to compare how architecture choice affects the resulting communities and codelength.

The project consists of the following files and folders:
- **test_graphs/**: folder containing small test graphs exported as GraphML, (weighted/unweighted, directed/undirected); as well as visualizations of each graph as .pdf
- **src/**: folder containing the implemented functions for infomap
  - `map_equation.py`: functions for computing & updating description length via map equation.
  - `optimize.py`:  functions for the optimization algorithm to search for an optimal community partition.
  - `utils.py`: helper functions used to generate & visualise test networks, load WikiCS, and write result files.
  - `neuromap.py`: helper functions & classes for the Neuromap community detection, taken from the [official Neuromap repository](https://github.com/chrisbloecker/neuromap).
- **notebooks/**: folder containing the notebooks used to test and compare the different components of the Infomap implementation, using igraph
  - `Test_networks.ipynb`: Jupyter notebook containing the SBM-based graph generation function used to generate the test graphs.
  - `Test_infomap.ipynb`: Jupyter notebook testing a self-implemented version of the Infomap description length computation via map equation, as well as custom update functions.
  - `SearchAlgorithm.ipynb`: Jupyter notebook for testing the functions needed for implementing the search algorithm to find the optimal community partition wrt. description length.
  - `SubmoduleRefinement.ipynb`: Jupyter notebook testing the recursive submodule refinement step and the full infomap workflow, validated against igraph's `community_infomap` and against the ground-truth partition.
  - `SingleNodeTests.ipynb`: Single-node edge-case tests for the custom Infomap implementation.
  - `uniform.ipynb`: map equation with uniform recorded teleportation (matches Laura's setup, sanity check).
  - `non-uniform.ipynb`: map equation with smart unrecorded teleportation (tutorial's recommended scheme).
  - `WikiCS_analysis.ipynb`: Jupyter notebook for loading and plotting the results of infomap/neuromap community detection on the WikiCS dataset.
  - `benchmark_infomap.ipynb`: testing and timing three Infomap implementations (custom, igraph, official) on Cora and CoraML over 10 trials.
  - `benchmark_neuromap.ipynb`: testing and timing four GNN architectures (GCN, GIN, GAT, GraphSAGE) for Neuromap on Cora and CoraML over 10 trials.
- `run_infomap_trials.py`: python script to run different infomap implementations on the WikiCS dataset.
- `run_neuromap.py`: python script to run neuromap with different GNN architectures on the WikiCS dataset.

