# Information Theory and Inference project: Maps of random walks on complex networks reveal community structure
Course project for the course *Information Theory and Inference* at the University of Padova, A.Y. 2025/2026. The project is based on the paper ["Maps of random walks on complex networks reveal community structure"](https://doi.org/10.1073/pnas.0706851105). Details to be added.

Contributors:
- Marco Foster ([@MarcoTFoster](https://github.com/MarcoTFoster))
- Laura Schulze ([@lm-schulze](https://github.com/lm-schulze))
- Savina Tsichli ([@savinats](https://github.com/savinats))
  
The project consists of the following files and folders (to be updated as the project progresses):
- **filename**: file description.

## Files in **_saviti_** branch

- **test_graphs/**: folder containing test graphs exported as GraphML, (weighted/unweighted, directed/undirected); as well as visualizations of each graph as .pdf
- **src/**: folder containing the implemented functions for infomap
  - `infomap_funcs.py`: Contains all helper functions used to generate & visualise test networks, compute & update description length, and perform the search for an optimal community partition.
  - `map_equation.py`: function definitions shared between the notebooks
- **notebooks/**: folder containing the notebooks used to test and compare the different components of the infomap implementation
  - `Test_networks.ipynb`: Jupyter notebook containing the SBM-based graph generation function used to generate the test graphs. Uses the igraph library.
  - `Test_infomap.ipynb`: Jupyter notebook testing a self-implemented version of the Infomap description length computation via map equation, as well as custom update functions. Uses the igraph library.
  - `SearchAlgorithm.ipynb`: Jupyter notebook for testing the functions needed for implementing the search algorithm to find the optimal community partition wrt. description length. Uses the igraph library.
  - `uniform.ipynb`: map equation with uniform recorded teleportation (matches Laura's setup, sanity check)
  - `nonuniform.ipynb`: map equation with smart unrecorded teleportation (tutorial's recommended scheme)
  - `summary.ipynb`: quick-reference notebook showing the final results from both implementations side by side
  - `bif.ipynb`: first attempt at the implementation of a basic infomap function
