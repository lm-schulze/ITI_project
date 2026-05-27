# library imports
import igraph as ig
import numpy as np
import random
import warnings

def generate_sbm(n, c, p_in, p_out, directed=False, weighted=False):
    """
    Generate a Stochastic Block Model graph.

    Parameters
    ----------
    n        : total number of nodes
    c        : number of communities
    p_in     : within-community connection probability
    p_out    : between-community connection probability
    directed : whether the graph should be directed
    weighted : whether the graph should be weighted 
               (if True, assigns uniform weights from 0 to 1 to edges)

    Returns
    -------
    g : igraph.Graph
    """

    # determine module sizes (distributing nodes as evenly as possible)
    module_sizes = np.full(c, n // c, dtype=int)
    module_sizes[:n % c] += 1   # distribute remainder evenly
    
    # build preference matrix for SBM
    pref_matrix = np.full((c, c), p_out)
    np.fill_diagonal(pref_matrix, p_in)

    # generate graph via igraph's SBM generator
    g = ig.Graph.SBM(
            pref_matrix.tolist(),
            module_sizes.tolist(),
            directed=directed,
            allowed_edge_types='simple'
        )  
    
    # explicitely assign community membership as vertex attribute
    g.vs["community"] = np.repeat(np.arange(c), module_sizes)

    # if weighted, assign random weights to edges
    if weighted:
        g.es["weight"] = np.random.rand(g.ecount())

    return g

# little helper for plotting purposes
# might make a a seperate python file for all utility functions later on
def visual_community_colors(g: ig.Graph, communities=None, skipLayout=False):
    """Creates dict of plotting arguments for visualising networks with community assignments.
    Nodes are colored according to their community, edge width is proportional to edge weights
    if the network is weighted. Community assignments can either be explicitly passed via the 
    "communities" argument or as vertex attribute "community" of input graph g. Basic usage for
    plotting an igraph.Graph g:
    ig.plot(g, **visual_community_colors(g))

    Args:
        g (ig.Graph): Input graph to plot.
        communities (list[int], optional): Explicit community assignment for each node. Defaults 
                to None.
        skipLayout (bool, optional): If True, layout is not specified in the resulting dict. Use
                if you want to specify the layout seperately, e.g. to be consistent for multiple
                plots of the same network. Defaults to False.

    Returns:
        dict: Dict of visual style settings to be passed to igraph.plot()
    """
    if communities is None and "community" in g.vertex_attributes():
        communities = g.vs["community"]
    c = max(communities) + 1
    palette = ig.RainbowPalette(n=c)

    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_color"] = [palette.get(i) for i in communities] if communities is not None else "lightblue"
    visual_style
    if not skipLayout:
        visual_style["layout"] = g.layout('fr')
    visual_style["bbox"] = (400, 400)
    visual_style["margin"] = 20
    visual_style["vertex_label_angle"] = 90
    visual_style["vertex_label_dist"] = 2.
    max_weight = max(g.es["weight"]) if "weight" in g.edge_attributes() else 1
    visual_style["edge_width"] = [1+ 5*w/max_weight for w in g.es["weight"]] if "weight" in g.edge_attributes() else 1
    visual_style["edge_color"] = "rgba(1,1,1,0.7)" if "weight" in g.edge_attributes() else "rgba(1,1,1,1)"
    return visual_style

def compute_exit_weights(g: ig.Graph, communities: list[int]) -> np.ndarray:
    """Compute community exit weights for a given undirected graph and community partition.
       Helper function for the description length computation via map equation.

    Args:
        g (ig.Graph): (Undirected) input graph.
        communities (list[int]): List of non-overlapping community labels for all nodes of input graph G.

    Returns:
        np.ndarray: Exit weights for each community.
    """
    weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))
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
    weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))
    edges = np.array(g.get_edgelist(), dtype=int) # array of edges

    src = edges[:, 0] # community of source node for each edge
    trg = edges[:, 1] # community of target node for each edge
    src_com = communities[src] # community of source node for each edge
    trg_com = communities[trg] # community of target node for each edge
    betw_communities = src_com != trg_com # true if edge connects different communities

    # exit flow on each edge:
    #flow = p[src] * weights / out_strength[src] # flow on each edge, proportional to node visit frequency and edge weight
    # handle out_strength = 0 case
    flow = np.where(out_strength[src] > 0, p[src] * weights / out_strength[src], 0.0)  # dangling → 0 flow

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
    safe_x = np.where(x > 0.0, x, 1.0)   # replace 0s with 1 to avoid that pesky Divide By 0 issue
    return np.where(x > 0.0, safe_x * np.log2(safe_x), 0.0) # set these points manually to 0

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

    communities = np.array(communities, dtype=int)
    num_communities = max(communities) + 1
    # number of communities in the partition
    N = g.vcount() # number of nodes in the graph

    # handle the edge-case (hehe) of a graph without edges, or nodes:
    if g.ecount() == 0 or N == 0: # graph doesn't have edges or nodes:
        if returnTerms:
            return 0.0, np.zeros(N), np.zeros(num_communities), np.zeros(num_communities)
        else:   
            return 0.0
    
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
            
    else:
        weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))
        total_weight_x2 = 2 * np.sum(weights) # total weight of all edges (x2 for undirected graphs)

        # compute ergodic node visit frequencies
        p = np.array(g.strength(weights="weight" if g.is_weighted() else None)) / total_weight_x2
    
        p_mod = np.zeros(num_communities) # initialise module visit frequency array
        np.add.at(p_mod, communities, p) # sum node visit frequencies for each community
        
        # compute module exit probabilities
        exit_weights = compute_exit_weights(g, communities) 
        q_mod = exit_weights / total_weight_x2

    q_sum = np.sum(q_mod) # total exit probability  
    p_loop = p_mod + q_mod

    if g.is_directed():
        exit_data = exit_flow
    else:
        exit_data = exit_weights

    if verbose:
        # diagnostics:
        print("p sum:        ", p.sum())
        print("p_mod sum:    ", p_mod.sum())       # should equal p.sum() = 1
        print("exit_data sum:", exit_data.sum())   # should be < 1
        print("q_mod sum:    ", q_mod.sum())       # should be < 1
        print("p_loop sum:   ", p_loop.sum())      # should be > 1 (= 1 + q_sum)
        print("any nan/inf:", np.any(~np.isfinite(q_mod)), np.any(~np.isfinite(p)))
        
    # compute via map equation
    L = safe_xlogx(q_sum) - 2*np.sum(safe_xlogx(q_mod)) \
        - np.sum(safe_xlogx(p)) + np.sum(safe_xlogx(p_loop))    

    if returnTerms:
        return L, p, p_mod, exit_data
    else:   
        return L
    

def update_exit_weights(g: ig.Graph, communities_old: list[int], exit_weights_old: np.ndarray,
                           node: int, comm_src: int, comm_trg: int) -> np.ndarray:
    """Update the exit weights incrementally when a single node is moved from its community to a target community.
       This is more efficient than recomputing from scratch for undirected graphs.

    Args:
        g (ig.Graph): (Undirected) input graph.
        communities_old (list[int]): List of non-overlapping community labels for all nodes.
        exit_weights_old (np.ndarray): Exit weights before the move.
        node (int): Node to move.
        comm_src (int): Source community of the node.
        comm_trg (int): Target community of the node.

    Returns:
        np.ndarray: Updated exit weights for each community.
    """
    
    communities = np.array(communities_old)
    # Safety checks
    if communities[node] != comm_src:
        raise ValueError(f"Node {node} is not in source community {comm_src}")
    if comm_src == comm_trg:
        return exit_weights_old.copy() 
    
    # Compute total degree of the node
    total_degree = g.strength(node, weights="weight" if g.is_weighted() else None)
    
    # Compute sum of weights to source and target communities
    W_src = 0.0
    W_trg = 0.0
    for neighbor in g.neighbors(node):
        comm = communities[neighbor]
        weight = g.es[g.get_eid(node, neighbor)]["weight"] if g.is_weighted() else 1.0
        if comm == comm_src:
            W_src += weight
        elif comm == comm_trg:
            W_trg += weight
    
    # Compute deltas
    delta_src = 2 * W_src - total_degree
    delta_trg = total_degree - 2 * W_trg
    
    # Update exit weights
    exit_weights_new = exit_weights_old.copy()
    exit_weights_new[comm_src] += delta_src
    exit_weights_new[comm_trg] += delta_trg
    
    return exit_weights_new


# it's a bit funkier when we're dealing with directed networks:
def update_exit_flow(g: ig.Graph, communities_old: list[int], p: np.ndarray, exit_flow_old: np.ndarray, node: int, comm_src: int, comm_trg: int) -> np.ndarray:
    """Update the community exit flow for a directed graph when one node changes communities.

    This function updates the exit flow incrementally instead of recomputing it from scratch.

    Args:
        g (ig.Graph): Directed input graph.
        communities_old (list[int]): List of non-overlapping community labels for all nodes.
        p (np.ndarray): Node visit frequencies.
        exit_flow_old (np.ndarray): Exit flow per community before the move.
        node (int): Node to move.
        comm_src (int): Source community of the moved node.
        comm_trg (int): Target community of the moved node.

    Returns:
        np.ndarray: Updated exit flow for each community.
    """

    communities = np.array(communities_old)
    # some safety checks
    if communities[node] != comm_src:
        raise ValueError(f"Node {node} is not in source community {comm_src}")
    if comm_src == comm_trg: # no change in community assignment 
        return exit_flow_old.copy() # exit flow stays the same 

    exit_flow = np.array(exit_flow_old, copy=True) # old exit flow
    # for flow computation we need the edge weights and outgoing strength
    weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64)) # network edge weights
    out_strength = np.array(g.strength(mode="out", weights="weight" if g.is_weighted() else None)) 
    
    node_out_strength = out_strength[node] # strength of outgoing links of moved node
    node_p = p[node] # visit frequency of moved node

    edges = np.array(g.get_edgelist(), dtype=int)
    out_edge_ids = np.array(g.incident(node, mode="out"), dtype=int)  # outgoing edges of moved node
    in_edge_ids = np.array(g.incident(node, mode="in"), dtype=int)  # incoming edges of moved node

    # remember, for the exit flow of a community we need consider its outgoing links
    # moving the node to another community affects the exit flows of comm_src and comm_trg 
    # for the other communities the assignment of node doesn't matter because it's external either way
    # so it contributes to the exit flow the same way as before

    # Update exit flow for outgoing edges from the moved node.
    if out_edge_ids.size > 0:
        out_edges = edges[out_edge_ids]
        trg = out_edges[:, 1] # get target nodes of outgoing edges
        trg_com = communities[trg] # get communities of target nodes
        flow = node_p * weights[out_edge_ids] / node_out_strength # compute the flow for each outgoing edge

        # Remove old contributions from comm_src and add new contributions to comm_trg.
        old_exit = np.sum(flow[trg_com != comm_src])  # old exit flow contribution for source comm.
        new_exit = np.sum(flow[trg_com != comm_trg])  # new exit flow contribution for target comm.
        exit_flow[comm_src] -= old_exit # subtract flow contribution from source community
        exit_flow[comm_trg] += new_exit # add flow contribution to target community

    # Update exit flow for incoming edges into the moved node from other nodes.
    # Only sources from comm_src or comm_trg can change whether they are external.
    # For incoming links from other communities it doesn't matter, as they will be external either way
    if in_edge_ids.size > 0:
        in_edges = edges[in_edge_ids]
        src = in_edges[:, 0]  # get source nodes of incoming edges
        src_com = communities[src] # get communities of source nodes
        flow_in = p[src] * weights[in_edge_ids] / out_strength[src] # compute the flow for each incoming edge

        # Edges from comm_src to the moved node become external after the move.
        # because they now connect comm_src and comm_trg
        # so they contribute to comm_src's exit flow
        mask_src = src_com == comm_src
        if np.any(mask_src):
            exit_flow[comm_src] += np.sum(flow_in[mask_src])

        # Edges from comm_trg to the moved node become internal after the move.
        # so they now longer contribute to the exit flow of comm_trg
        mask_trg = src_com == comm_trg
        if np.any(mask_trg):
            exit_flow[comm_trg] -= np.sum(flow_in[mask_trg])

    return exit_flow

def update_node_move_description_length(g: ig.Graph, communities_old: list[int], p_old: np.ndarray, p_mod_old: np.ndarray, exits_old: np.ndarray, node: int, comm_trg: int, tau: float = 0.15, returnTerms: bool = False, verbose: bool = False) -> float:
    """Compute the change in description length if a single node is moved from its community to a different community.
        This can be used for search algorithms that iteratively move nodes between communities to improve the partitioning.

    Args:
        g (ig.Graph): input graph (can be directed or undirected, weighted or unweighted)
        communities_old (list[int]): List of non-overlapping community labels for all nodes before the merge.
        p_old (np.ndarray): The old node visit frequencies.
        p_mod_old (np.ndarray): The old module visit frequencies.
        exits_old (np.ndarray): The old exit flows/weights.
        node (int): Node to be moved to a different community.
        comm_trg (int): Target community to move the node to.
        tau (float, optional): Teleportation probability for directed graphs. Defaults to 0.15.

    Returns:
        float: The new description length after moving the node.
    """
    
    # get the community the original node belongs to
    comm_src = communities_old[node]
    if verbose: 
        print(f"Moving node {node} from community {comm_src} to {comm_trg}")
    # if source community == target community: no change in description length.
    if comm_src == comm_trg:
        warnings.warn(f"Node already in target community {comm_trg}! No change in description length.")
        if returnTerms:
            return None, communities_old, exits_old 
        else:
            return None
    
    communities_old = np.array(communities_old)
    communities_new = communities_old.copy()
    communities_new[node] = comm_trg
    
    num_communities = len(np.unique(communities_new)) #  TODO: handle case of empty community later!
    N = g.vcount()
    
    # Update p_mod
    # subtract visit frequency p of node from source community
    # and add it to target community
    p_node = p_old[node] # get visit frequency of moved node

    if comm_trg not in communities_old: # we're
        needed_size = comm_trg + 1  # comm_trg might be a new singleton label
    else: 
        needed_size = num_communities # otherwise we just need to make sure we have enough space for the existing communities
    
    if needed_size > len(p_mod_old):
        p_mod_new = np.zeros(needed_size)
        p_mod_new[:len(p_mod_old)] = p_mod_old
    else:
        p_mod_new = p_mod_old.copy()

    if needed_size > len(exits_old):
        exits_new = np.zeros(needed_size)
        exits_new[:len(exits_old)] = exits_old
    else:
        exits_new = exits_old.copy()

    p_mod_new[comm_src] -= p_node  # subtract from source community
    p_mod_new[comm_trg] += p_node  # add to target community
    
    if g.is_directed():
        # Update node counts
        node_counts = np.bincount(communities_new, minlength=num_communities)
        
        # Update exit flows
        exit_flow_new = update_exit_flow(g, communities_old, p_old, exits_new, node, comm_src, comm_trg)
        
        # Compute q_mod
        q_mod = tau * (N - node_counts) / N * p_mod_new + (1 - tau) * exit_flow_new
            
    else:
        # For undirected, total_weight_x2 is constant
        weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))
        total_weight_x2 = 2 * np.sum(weights)
        
        # Update exit weights
        exit_weights_new = update_exit_weights(g, communities_old, exits_new, node,
                                               comm_src, comm_trg)
        # Compute q_mod
        q_mod = exit_weights_new / total_weight_x2

    q_sum = np.sum(q_mod) # total exit probability  
    p_loop = p_mod_new + q_mod

    if g.is_directed():
        exit_data = exit_flow_new
    else:
        exit_data = exit_weights_new

    if verbose:
        # diagnostics:
        print("p sum:        ", p_old.sum())
        print("p_mod sum:    ", p_mod_new.sum())       # should equal p.sum() = 1
        print("exit_data sum:", exit_data.sum())   # should be < 1
        print("q_mod sum:    ", q_mod.sum())       # should be < 1
        print("p_loop sum:   ", p_loop.sum())      # should be > 1 (= 1 + q_sum)
        print("any nan/inf:", np.any(~np.isfinite(q_mod)), np.any(~np.isfinite(p_old)))
        
        
    # compute via map equation
    L = safe_xlogx(q_sum) - 2*np.sum(safe_xlogx(q_mod)) \
        - np.sum(safe_xlogx(p_old)) + np.sum(safe_xlogx(p_loop))    
    

    if returnTerms:     
        return L, communities_new, p_mod_new, exit_data

    else:   
        return L


def node_movement_optimization(g, initial_communities=None, returnTerms=False, verbose=False):
    """Optimize community assignment of single nodes by sequentially iterating through them
    in a random order and assigning them to the neighbouring community that yields the greatest
    decrease in description length (or leaving them if current community yields lowest description length). 
    Repeats until no further improving node moves are possible. Corresponds to Phase 1 of the optimization 
    algorithm.

    Args:
        g (igraph.Graph): Input graph. Also supports directed and/or weighted graphs.
        initial_communities(list[int], optional): Initial community assignment to start the optimization 
                    from. If None, optimization starts with each node assigned to its own community.   
        returnTerms (bool, optional): Whether to return additional information besides best community
                    assignment. Defaults to False.
        verbose (bool, optional): Whether to print info for debugging. Defaults to False.

    Returns:
        list[int]: List of best community assignments found. If returnTerms is True, also returns 
                   description length L and community exit flows/weights of current structure.
    """
    nodes = g.vs.indices # get list of nodes
    N_nodes = g.vcount()
    neighborhood = g.neighborhood(mindist=1) # get list of neighbours for all nodes
    #neighborhood = [np.array(nbs) for nbs in neighborhood] # convert to list of numpy arrays for easier indexing

    # initialize community partition with each node being its own community
    if initial_communities is None:
        communities = np.arange(N_nodes) # start with each node assigned to its own community
    else: 
        communities = initial_communities.copy()

    # compute description length including some intermediate terms:
    L, p, p_mod, exit_data = compute_description_length(g, communities, returnTerms=True)

    if verbose:
        print(f"Starting from description length: {L}")

    optimizable=True
    while optimizable: # while there are still improvements via node moves:
        # randomize node sequence
        random.shuffle(nodes)

        # track how many nodes remain in their og community
        no_move_ctr = 0

        # for each node go through neighbours (if different community(?))
        for n in nodes:
            neighbors = neighborhood[n] # get neighbors of node
            nb_comms = communities[neighbors] # get communties of neighbors
            src_comm = communities[n] # community the current node is in
            comms_to_test = np.unique(nb_comms) # get unique neighbor communities
            comms_to_test = comms_to_test[comms_to_test != src_comm] # remove node's own community from communities to test
             # also give the option of moving to a new singleton community.
            fresh_singleton = int(np.max(communities) + 1)  # guaranteed unused label
            comms_to_test = np.append(comms_to_test, fresh_singleton)   

            L_best, communities_best, p_mod_best, exit_data_best = L, communities.copy(), p_mod.copy(), exit_data.copy() 
            # go through unique neighbouring communities  
            for nbc in comms_to_test:
                # get new description length for assigning node to different community
                L_new, communities_new, p_mod_new, exit_data_new = update_node_move_description_length(g, communities, p, p_mod, exit_data, n, nbc, returnTerms=True, verbose=verbose)
                if L_new is not None and L_new < L_best: # if better description length
                    # update best constellation
                    L_best, communities_best, p_mod_best, exit_data_best = L_new, communities_new.copy(), p_mod_new.copy(), exit_data_new.copy()
            
            # check if a change has been made
            if communities[n] == communities_best[n]: 
                no_move_ctr += 1

            # take over the new best community partition & data (might be identical with old one)
            L, communities, p_mod, exit_data = L_best, communities_best.copy(), p_mod_best.copy(), exit_data_best.copy()

            # relabel to contiguous, 0-indexed labels
            _, communities = np.unique(communities, return_inverse=True)
            L, p, p_mod, exit_data = compute_description_length(
                g, communities, returnTerms=True)
        # only stop optimizing if not a single improving move has been made in the sequence
        # otherwise keep optimizing
        optimizable = no_move_ctr < N_nodes 

        if verbose:
            print(f"Current best description length: {L_best}")
            print(f"Number of nodes that have been moved this iteration: {N_nodes-no_move_ctr}")
            if optimizable:
                print("Continuing optimization.")
            else: 
                print("Optimization finished!")

    # relabel to obtain contiguous community labels
    _, communities = np.unique(communities, return_inverse=True)
    L, p, p_mod, exit_data = compute_description_length(
            g, communities, returnTerms=True
        )

    if verbose:
        print(f"Final number of communities: {len(np.unique(communities))}")
        print(f"Final description length: {L}")

    if returnTerms:
        return communities, L, p_mod, exit_data
    else:
        return communities
    
    

def compress_network(g: ig.Graph, communities: list[int], verbose=False) -> tuple:
    """
    Implements Phase 2 of the infomap search algorithm. Compresses the network by
    collapsing each community into a single super-node and aggregating edge weights,
    with within-community edges resulting in self-loops.

    Args:
    g (ig.Graph): Input graph. Supports directed/undirected and weighted/unweighted.
    communities (list[int] or np.ndarray): Community label for each node of g. Labels
        need not be 0-indexed or contiguous (e.g. after previous merge steps some 
        labels may be absent from the range).
    verbose (bool, optional): Whether to print verbose output for debugging. Defaults to False.

    Returns:
    g_compressed (ig.Graph): Compressed graph with len(np.unique(communities)) nodes.
        Always weighted (aggregated weights stored as the "weight" edge attribute).
        Directedness matches the input graph. May contain self-loops.
    community_map (np.ndarray): Sorted array of the unique original community labels,
        where community_map[i] is the original label of super-node i in
        g_compressed. Because the array is sorted, np.searchsorted can
        cheaply convert original community labels to compressed-node indices.
    """
    communities = np.array(communities)

    # --- Get 0-indexed node IDs for communities -------------------------
    # The plan is to get a sorted list of the unique communities, and have the node
    # indices of the compressed graph correspond to the list indices of the corresponding
    # community in the sorted list. The list will be returned alongside the compressed Graph
    # to allow recovery of original community assignments
    unique_communities = np.unique(communities) # get all unique community labels, sorted
    n_communities = int(len(unique_communities)) # get number of communities

    # For each original node, get the position of its community label in the sorted
    # unique_communities array 
    node_to_compressed = np.searchsorted(unique_communities, communities)
    # basically contains for each node the index of the community instead of the community label
    # these indices will be the supernode indices of the compressed graph

    if verbose:
        print(f"Input graph has {n_communities} unique communities, {g.vcount()} nodes and {g.ecount()} edges.")


    # --- Build compressed edge list with aggregated weights -----------------
    # We'll basically build a graph with a number of nodes = number of communities
    # and then insert the correctly aggregated edges that we compute here
    if g.ecount() > 0: # if we have edges get the weights
        weights = np.array(
            g.es["weight"] if g.is_weighted() else np.ones(g.ecount()),
            dtype=np.float64
        )
        edges = np.array(g.get_edgelist(), dtype=np.int64) # build edgelist

        # map each start/endpoint to its compressed-graph node index
        # so basically instead of (starting node, ending node) we now have
        # the community indices (starting community, ending community)
        new_src = node_to_compressed[edges[:, 0]].astype(np.int64)  
        new_trg = node_to_compressed[edges[:, 1]].astype(np.int64)

        # Encode each (src, trg) pair as a single int64 key for O(E log E)
        # aggregation via np.unique instead of a Python dict loop.
        # with this, basically src = edge_key // n_communities, 
        # trg = edge_key % n_communities
        edge_keys = new_src * np.int64(n_communities) + new_trg
        # with this, any edges connecting the same communities a and b will have the same edge key
        # which we can then use to aggregate the weights

        # Sum weights of all edges that map to the same (src, trg) pair.
        # First, np.unique gives the unique keys and an inverse mapping; 
        unique_keys, inverse_idx = np.unique(edge_keys, return_inverse=True)
        # gets sorted unique edge keys, and a list containing for each edge_key (so for each edge)
        # in the original list the index of the key in the unique_keys list (inverse mapping)

        # np.add.at accumulates weights into the correct bucket in one vectorised pass.
        agg_weights = np.zeros(len(unique_keys), dtype=np.float64) # init array for edge weight aggregation
        np.add.at(agg_weights, inverse_idx, weights) 
        # adds the weights of each edge to the element in agg_weights whose index corresponds
        # to the inverse_idx of that edge, which is the same as the index of the unique keys
        # so for any edges connecting the same communities a and b (who will have the same 
        # edge_key, and thus the same inverse_idx), the weights are summed.

        # Decode integer keys back to (src, trg) pairs
        # keeping only the unique ones
        compressed_src = (unique_keys // n_communities).tolist() 
        compressed_trg = (unique_keys %  n_communities).tolist()
        new_edges = list(zip(compressed_src, compressed_trg)) 

    else:                       # original graph has no edges
        new_edges = []
        agg_weights = np.array([], dtype=np.float64)

    # --- Assemble the compressed igraph.Graph --------------------------------
    # self-loops from intra-community edges are explicitly required here and are handled
    # correctly by compute_description_length (they satisfy src_com == trg_com
    # and are therefore excluded from exit weights/flows by those helpers).
    # GOD I HOPE THAT'S ACTUALLY TRUE 

    if verbose:
        print(f"Creating compressed graph with {n_communities} nodes, {len(new_edges)} aggregated edges.")

    # create graph with # nodes = # communities of g
    g_compressed = ig.Graph(n=n_communities, directed=g.is_directed())  
    if new_edges: # if we have any edges to add
        g_compressed.add_edges(new_edges) # add the new aggregated edges
        g_compressed.es["weight"] = agg_weights.tolist() # assign them the aggregated weights
    
    # Return a copy so callers cannot accidentally mutate the internal array
    # we should be able to reconstruct the assignments from the unique_communites list
    # as it contains the mapping of og community -> compressed node index (== list index)
    return g_compressed, unique_communities.copy()

