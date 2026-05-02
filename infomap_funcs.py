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


def compute_description_length(g: ig.Graph, communities: list[int], tau: float = 0.15, returnTerms: bool = False, verbose: bool = False) -> float :
    """Compute the description length of a given partitioning of the graph using the map equation.
        Supports both directed and undirected graphs.

    Args:
        g (ig.Graph): input graph (can be directed or undirected, weighted or unweighted)
        communities (list[int]): List of non-overlapping community labels for all nodes of input graph G.
        tau (float, optional): Teleportation probability for directed graphs. Defaults to 0.15.
        returnTerms (bool, optional): Whether to return additional terms that may be useful for updating later.
        verbose (bool, optional): Whether to print diagnostic information for debugging. Defaults to False.

    Returns:
        float: Description length of the given partitioning of the graph according to the map equation.
        or i returnTerms=True, additionally returns the p, p_mod, exit_flow/weights

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
        exit_weights = compute_exit_weights(g, communities) 
        q_mod = exit_weights / total_weight_x2

    q_sum = np.sum(q_mod) # total exit probability  
    p_loop = p_mod + q_mod
        
    # compute via map equation
    L = safe_xlogx(q_sum) - 2*np.sum(safe_xlogx(q_mod)) \
        - np.sum(safe_xlogx(p)) + np.sum(safe_xlogx(p_loop))    

    if returnTerms:

        if g.is_directed():
            return L, p, p_mod, exit_flow 
        else:
            return L, p, p_mod, exit_weights
    else:   
        return L


def update_exit_weights(g: ig.Graph, communities_old: list[int], exit_weights_old: list[int], comm1: int, comm2:int ) -> np.ndarray:
    """ Compute the change in exit weights & update if 2 communities are merged.
        This can be used for search algorithms that iteratively merge communities to improve the partitioning.


    Args:
        g (ig.Graph): (Undirected) input graph.
        communities_old (list[int]): List of non-overlapping community labels for all nodes of input graph G.
        comm1 (int): First community to merge.
        comm2 (int): Second community to merge.
        exit_weights_old (list[int]): Exit weights before merging.

    Returns:
        np.ndarray: Exit weights for each community after merging.
    """
    
    if comm1 == comm2:
        raise ValueError("Cannot merge a community with itself")
    
    # Ensure comm1 < comm2
    if comm1 > comm2:
        comm1, comm2 = comm2, comm1
    
    communities = np.array(communities_old)
    exit_weights_old = np.array(exit_weights_old)
    weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
    edges = np.array(g.get_edgelist(), dtype=int)
    src_com = communities[edges[:, 0]]
    trg_com = communities[edges[:, 1]]
    
    # Edges between comm1 and comm2
    betw_12 = ((src_com == comm1) & (trg_com == comm2)) | ((src_com == comm2) & (trg_com == comm1))
    inter_weight = np.sum(weights[betw_12])
    
    # New exit weight for merged community
    new_exit_merged = exit_weights_old[comm1] + exit_weights_old[comm2] - 2 * inter_weight
    
    # Create new exit weights array, remove comm2 and shift
    exit_weights_updated = np.delete(exit_weights_old, comm2)
    exit_weights_updated[comm1] = new_exit_merged
    
    return exit_weights_updated

# it's a bit funkier when we're dealing with directed networks:
def update_exit_flow(g: ig.Graph, communities_old: list[int], p: np.ndarray, exit_flow_old: np.ndarray, comm1: int, comm2: int) -> np.ndarray:
    """Compute the change in community exit flow if 2 communities are merged.
    This can be used for search algorithms that iteratively merge communities to improve the partitioning.

    Args:
        g (ig.Graph): (Directed) input graph.
        communities_old (list[int]): List of non-overlapping community labels for all nodes before merging.
        p (np.ndarray): Node visit frequencies.
        exit_flow_old (np.ndarray): Exit flows before merging.
        comm1 (int): First community to merge.
        comm2 (int): Second community to merge.

    Returns:
        np.ndarray: Exit flow for each community after merging.
    """
    
    if comm1 == comm2:
        raise ValueError("Cannot merge a community with itself")
    
    # Ensure comm1 < comm2
    if comm1 > comm2:
        comm1, comm2 = comm2, comm1
    
    communities = np.array(communities_old)
    out_strength = np.array(g.strength(mode="out", weights="weight" if g.is_weighted() else None))
    weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
    edges = np.array(g.get_edgelist(), dtype=int)
    src = edges[:, 0]
    trg = edges[:, 1]
    src_com = communities[src]
    trg_com = communities[trg]
    
    # Edges between comm1 and comm2
    betw_12 = ((src_com == comm1) & (trg_com == comm2)) | ((src_com == comm2) & (trg_com == comm1))
    
    # Flow on those edges (only from src)
    flow_inter = p[src[betw_12]] * weights[betw_12] / out_strength[src[betw_12]]
    inter_flow_sum = np.sum(flow_inter)
    
    # New exit flow for merged community
    new_exit_merged = exit_flow_old[comm1] + exit_flow_old[comm2] - inter_flow_sum
    
    # Create new exit flow array, remove comm2 and shift
    exit_flow_updated = np.delete(exit_flow_old, comm2)
    exit_flow_updated[comm1] = new_exit_merged
    
    return exit_flow_updated


def update_description_length(g: ig.Graph, communities_old: list[int], p_old: np.ndarray, p_mod_old: np.ndarray, exits_old: np.ndarray, comm1: int, comm2: int, tau: float = 0.15, returnTerms: bool = False, verbose: bool = False) -> float:
    """Compute the change in description length if 2 communities are merged.
        This can be used for search algorithms that iteratively merge communities to improve the partitioning.

    Args:
        g (ig.Graph): input graph (can be directed or undirected, weighted or unweighted)
        communities_old (list[int]): List of non-overlapping community labels for all nodes before the merge.
        p_old (np.ndarray): The old node visit frequencies.
        p_mod_old (np.ndarray): The old module visit frequencies.
        exits_old (np.ndarray): The old exit flows/weights.
        comm1 (int): The first community to be merged.
        comm2 (int): The second community to be merged.
        tau (float, optional): Teleportation probability for directed graphs. Defaults to 0.15.

    Returns:
        float: The new description length after merging the communities.
    """

    if comm1 == comm2:
        raise ValueError("Cannot merge a community with itself")
    
    # Ensure comm1 < comm2
    if comm1 > comm2:
        comm1, comm2 = comm2, comm1
    
    communities_old = np.array(communities_old)
    communities_new = np.where(communities_old != comm2, communities_old, comm1)
    num_communities = len(p_mod_old)
    N = g.vcount()
    
    # Update p_mod
    p_mod_new = np.delete(p_mod_old, comm2)
    p_mod_new[comm1] += p_mod_old[comm2]
    
    if g.is_directed():
        # Update node counts
        node_counts_old = np.bincount(communities_old, minlength=num_communities)
        node_counts_new = np.delete(node_counts_old, comm2)
        node_counts_new[comm1] += node_counts_old[comm2]
        
        # Update exit flows
        exit_flow_new = update_exit_flow(g, communities_old, p_old, exits_old, comm1, comm2)
        
        # Compute q_mod
        q_mod = tau * (N - node_counts_new) / N * p_mod_new + (1 - tau) * exit_flow_new
        
        if verbose:
            print("p sum:        ", p_old.sum())
            print("p_mod sum:    ", p_mod_new.sum())
            print("exit_flow sum:", exit_flow_new.sum())
            print("q_mod sum:    ", q_mod.sum())
            print("any nan/inf:", np.any(~np.isfinite(q_mod)), np.any(~np.isfinite(p_old)))
            
    else:
        # For undirected, total_weight_x2 is constant
        weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
        total_weight_x2 = 2 * np.sum(weights)
        
        # Update exit weights
        exit_weights_new = update_exit_weights(g, communities_old, exits_old, comm1, comm2)
        
        # Compute q_mod
        q_mod = exit_weights_new / total_weight_x2

    q_sum = np.sum(q_mod) # total exit probability  
    p_loop = p_mod_new + q_mod
        
    # compute via map equation
    L = safe_xlogx(q_sum) - 2*np.sum(safe_xlogx(q_mod)) \
        - np.sum(safe_xlogx(p_old)) + np.sum(safe_xlogx(p_loop))    
    
    exit_data = exit_flow_new if g.is_directed() else exit_weights_new

    if returnTerms:
        return L, communities_new, p_old, p_mod_old, exit_data

    else:   
        return L