# library imports
import igraph as ig
import numpy as np
import warnings

def compute_exit_weights(g: ig.Graph, communities: list[int]) -> np.ndarray:
    """Compute community exit weights for a given undirected graph and community partition.
       Helper function for the description length computation via map equation.

    Args:
        g (ig.Graph): (Undirected) input graph.
        communities (list[int]): List of non-overlapping community labels for all nodes of input graph G.

    Returns:
        np.ndarray: Exit weights for each community.
    """
    weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
    communities = np.array(communities) # community membership list for each node
    exit_weights = np.zeros(max(communities) + 1) # initialise exit weight array

    edges = np.array(g.get_edgelist(), dtype=int) # array of edges
    src= communities[edges[:, 0]] # community of source node for each edge
    trg = communities[edges[:, 1]] # community of target node for each edge

    betw_communities = src != trg # true if edge connects different communities
    # for directed graphs, only consider outgoing edges for exit weights
    np.add.at(exit_weights, src[betw_communities], weights[betw_communities]) 
    if not g.is_directed():
        # for undirected graphs, consider both directions for exit weights
        np.add.at(exit_weights, trg[betw_communities], weights[betw_communities])

    return exit_weights

# it's a bit funkier when we're dealing with directed networks:
def compute_exit_flow(g: ig.Graph, communities: list[int], p: np.ndarray) -> np.ndarray:
    """Compute community exit flow for a given directed graph, community partition, and node visit frequencies.
       Helper function for the description length computation via map equation.

    Args:
        g (ig.Graph): (Directed) input graph.
        communities (list[int]): List of non-overlapping community labels for all nodes of input graph G.
        p (np.ndarray): Node visit frequencies.

    Returns:
        np.ndarray: Exit flow for each community.
    """

    communities = np.array(communities) # community membership list for each node
    out_strength = np.array(g.strength(mode="out", weights="weight" if g.is_weighted() else None)) # strength of outgoing links for each node
    weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
    edges = np.array(g.get_edgelist(), dtype=int) # array of edges

    src = edges[:, 0] # community of source node for each edge
    trg = edges[:, 1] # community of target node for each edge
    src_com = communities[src] # community of source node for each edge
    trg_com = communities[trg] # community of target node for each edge
    betw_communities = src_com != trg_com # true if edge connects different communities

    # exit flow on each edge:
    flow = p[src] * weights / out_strength[src] # flow on each edge, proportional to node visit frequency and edge weight
    exit_flow = np.zeros(max(communities) + 1) # initialise exit weight array
    np.add.at(exit_flow, src_com[betw_communities], flow[betw_communities]) 

    return exit_flow

    
# compute x*log2(x) and safely handle log(0) issues:
# by safely handle I mean just set it to zero
def safe_xlogx(x):
    """Compute x*log2(x) safely, setting log(0) to zero.

    Args:
        x (_type_): input value or array for which to compute x*log2(x)

    Returns:
        _type_: x*log2(x) for x > 0, and 0 for x <= 0
    """
    return np.where(x > 0, x * np.log2(x), 0.0)

# originally based off of the PageRank Wikipedia, hehe
# but changed to row-stochastic, and with dangling node handling
def pagerank(M, tau: float = 0.15, tol: float = 1e-15, maxiter: int = 1e6):
    """PageRank algorithm with teleportation probability tau. Returns ranking of nodes (pages) in the adjacency matrix.

    Parameters
    ----------
    M : numpy array
        adjacency/strength matrix where M[i,j] = weight of edge i -> j  (rows are sources)
    tau : float, optional
        teleportation probability, by default 0.15
    tol : float, optional
        tolerance for convergence, by default 1e-15
    maxiter : int, optional
        maximum number of iterations to prevent infinite loops, by default 1e6

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],

    """

    N = M.shape[0]
    row_sums = M.sum(axis=1)   # corrsponds to out strength
    dangling = (row_sums == 0) # dangling nodes (no outgoing edges)
    row_sums_safe = np.where(dangling, 1, row_sums) # set to one for normalisation
    M_normalised = M / row_sums_safe[:, None]   # row-stochastic: T[i,j] = p(i->j)

    p = np.ones(N) / N # init with uniform node visit prob
    for i in range(int(maxiter)):
        # dangling nodes redistribute uniformly
        dangling_sum = p[dangling].sum()
        p_new = (1 - tau) * (p @ M_normalised) + (1 - tau) * dangling_sum / N + tau / N
        if np.linalg.norm(p_new - p) < tol:
            return p_new
        p = p_new

    warnings.warn(f"PageRank did not converge after {maxiter} iterations.")
    return p


def compute_description_length(g: ig.Graph, communities: list[int], tau: float = 0.15, verbose: bool = False) -> float :
    """Compute the description length of a given partitioning of the graph using the map equation.
        Supports both directed and undirected graphs.

    Args:
        g (ig.Graph): input graph (can be directed or undirected, weighted or unweighted)
        communities (list[int]): List of non-overlapping community labels for all nodes of input graph G.
        tau (float, optional): Teleportation probability for directed graphs. Defaults to 0.15.
        verbose (bool, optional): Whether to print diagnostic information for debugging. Defaults to False.

    Returns:
        float: Description length of the given partitioning of the graph according to the map equation.
    """

    num_communities = max(communities) + 1 # number of communities in the partition
    N = g.vcount() # number of nodes in the graph
    
    if g.is_directed():
        # get adjacency matrix
        adj = np.array(g.get_adjacency(attribute="weight" if g.is_weighted() else None).data)
        # compute node visit frequencies with teleportation
        p = pagerank(adj, tau=tau)
        # compute module visit frequencies
        p_mod = np.zeros(num_communities)
        np.add.at(p_mod, communities, p) # sum node visit frequencies for each community
        # compute module exit probabilities (flow leaving modules due to inter-community edges)
        exit_flow = compute_exit_flow(g, communities, p)

        # compute module exit probabilities with teleportation correction:
        q_mod = tau * (N-np.bincount(communities, minlength=num_communities)) / N * p_mod \
            + (1-tau) * exit_flow
        if verbose:
            # diagnostics:
            print("p sum:        ", p.sum())
            print("p_mod sum:    ", p_mod.sum())       # should equal p.sum() = 1
            print("exit_flow sum:", exit_flow.sum())   # should be < 1
            print("q_mod sum:    ", q_mod.sum())       # should be < 1
            print("p_loop sum:   ", p_loop.sum())      # should be > 1 (= 1 + q_sum)
            print("any nan/inf:", np.any(~np.isfinite(q_mod)), np.any(~np.isfinite(p)))
            
    else:
        weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
        total_weight_x2 = 2 * np.sum(weights) # total weight of all edges (x2 for undirected graphs)

        # compute ergodic node visit frequencies
        p = np.array(g.strength(weights="weight" if g.is_weighted() else None)) / total_weight_x2
    
        p_mod = np.zeros(max(communities) + 1)
        np.add.at(p_mod, communities, p) # sum node visit frequencies for each community
        
        # compute module exit probabilities
        q_mod = compute_exit_weights(g, communities) / total_weight_x2

    q_sum = np.sum(q_mod) # total exit probability  
    p_loop = p_mod + q_mod
        
    # compute via map equation
    L = safe_xlogx(q_sum) - 2*np.sum(safe_xlogx(q_mod)) \
        - np.sum(safe_xlogx(p)) + np.sum(safe_xlogx(p_loop))    

    return L