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


def compute_description_length(g, communities, tau=0.15, 
                               teleportation="uniform",
                               returnTerms=False, verbose=False):
    """Compute the description length of a partition using the map equation.
    
    Args:
        g: input graph (directed/undirected, weighted/unweighted)
        communities: community label for each node
        tau: teleportation probability (only matters for directed graphs)
        teleportation: "uniform" (recorded) or "nonuniform" (smart unrecorded).
            For undirected graphs, this flag is ignored — teleportation has
            no effect on undirected results.
        returnTerms: also return intermediate terms (p, p_mod, exit_data)
        verbose: print diagnostic info
    """
    if teleportation not in ("uniform", "nonuniform"):
        raise ValueError(f"teleportation must be 'uniform' or 'nonuniform', got {teleportation!r}")
    
    communities = np.array(communities)

    # relabel just in case for 0-indexed, contiguous labels

    _, communities = np.unique(np.array(communities), return_inverse=True)
    num_communities = int(communities.max()) + 1 
    N = g.vcount()

    # handle the edge-case (hehe) of a graph without edges, or nodes:
    if g.ecount() == 0 or N == 0: # graph doesn't have edges or nodes:
        if returnTerms:
            return 0.0, np.zeros(N), np.zeros(num_communities), np.zeros(num_communities)
        else:   
            return 0.0
    
    if g.is_directed():
        adj = np.array(g.get_adjacency(attribute="weight" if g.is_weighted() else None).data, dtype=float)
        
        if teleportation == "uniform":
            # === Uniform recorded teleportation ===
            p = pagerank(adj, tau=tau)
            p_mod = np.zeros(num_communities)
            np.add.at(p_mod, communities, p)
            exit_flow = compute_exit_flow(g, communities, p)
            
            # q_mod includes the teleportation term
            n_mod = np.bincount(communities, minlength=num_communities)
            q_mod = tau * (N - n_mod) / N * p_mod + (1 - tau) * exit_flow
            
            # symmetric formula (same q for index and module codebook)
            q_sum = np.sum(q_mod)
            p_loop = p_mod + q_mod
            L = safe_xlogx(q_sum) - 2 * np.sum(safe_xlogx(q_mod)) \
                - np.sum(safe_xlogx(p)) + np.sum(safe_xlogx(p_loop))
            exit_data = exit_flow
            
        else:  # nonuniform
            # === Smart unrecorded teleportation ===
            p = pagerank_nonuniform(adj, tau=tau)
            p_mod = np.zeros(num_communities)
            np.add.at(p_mod, communities, p)
            exit_flow = compute_exit_flow(g, communities, p)
            enter_flow = compute_enter_flow_nonuniform(g, communities, p)
            
            # asymmetric formula: enter for index, exit for module
            q_enter = enter_flow
            q_exit = exit_flow
            q_enter_sum = np.sum(q_enter)
            p_loop = p_mod + q_exit
            L = safe_xlogx(q_enter_sum) - np.sum(safe_xlogx(q_enter)) \
                - np.sum(safe_xlogx(q_exit)) \
                - np.sum(safe_xlogx(p)) + np.sum(safe_xlogx(p_loop))
            exit_data = exit_flow
            q_mod = exit_flow   # for returnTerms/verbose compatibility
    
    else:
        # === Undirected case — same for both teleportation schemes ===
        weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))
        total_weight_x2 = 2 * np.sum(weights)
        p = np.array(g.strength(weights="weight" if g.is_weighted() else None)) / total_weight_x2
        
        p_mod = np.zeros(num_communities)
        np.add.at(p_mod, communities, p)
        
        exit_weights = compute_exit_weights(g, communities)
        q_mod = exit_weights / total_weight_x2
        
        q_sum = np.sum(q_mod)
        p_loop = p_mod + q_mod
        L = safe_xlogx(q_sum) - 2 * np.sum(safe_xlogx(q_mod)) \
            - np.sum(safe_xlogx(p)) + np.sum(safe_xlogx(p_loop))
        exit_data = exit_weights
    
    if verbose:
        print(f"teleportation: {teleportation}")
        print("p sum:        ", p.sum())
        print("p_mod sum:    ", p_mod.sum())
        print("exit_data sum:", exit_data.sum())
        print("q_mod sum:    ", q_mod.sum())
        print("p_loop sum:   ", p_loop.sum())
    
    if returnTerms:
        return L, p, p_mod, exit_data
    else:
        return L
    


def update_exit_weights(g: ig.Graph, communities_old: list[int], exit_weights_old: np.ndarray,
                        node: int, comm_src: int, comm_trg: int) -> np.ndarray:
    """Update exit weights incrementally when a single node moves communities.
    This is more efficient than recomputing from scratch for undirected graphs.

    Note: Self-loops are explicitly excluded: they never cross community boundaries
    and must not contribute to exit-weight deltas. 

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
    # safety checks
    if communities[node] != comm_src:
        raise ValueError(f"Node {node} is not in source community {comm_src}")
    if comm_src == comm_trg:
        return exit_weights_old.copy()

    incident_eids = np.array(g.incident(node), dtype=int)
    all_edges     = np.array(g.get_edgelist(), dtype=int)
    all_weights   = np.array(
        g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64)
    )

    inc_edges   = all_edges[incident_eids]
    inc_weights = all_weights[incident_eids]

    neighbor_nodes = np.where(inc_edges[:, 0] == node, inc_edges[:, 1], inc_edges[:, 0])

    # KEY FIX
    # Discard self-loops: they are always intra-community and never affect exit
    # weights regardless of which community the node is in.
    not_self    = neighbor_nodes != node
    neighbor_nodes = neighbor_nodes[not_self]
    inc_weights    = inc_weights[not_self]

    neighbor_comms = communities[neighbor_nodes]
    total_degree   = np.sum(inc_weights)          # non-self-loop degree only

    W_src = np.sum(inc_weights[neighbor_comms == comm_src])
    W_trg = np.sum(inc_weights[neighbor_comms == comm_trg])

    delta_src = 2 * W_src - total_degree          # now correct for self-loop nodes
    delta_trg = total_degree - 2 * W_trg          # now correct for self-loop nodes

    exit_weights_new = exit_weights_old.copy()
    exit_weights_new[comm_src] += delta_src
    exit_weights_new[comm_trg] += delta_trg
    return exit_weights_new


def update_exit_flow(g: ig.Graph, communities_old: list[int], p: np.ndarray,
                     exit_flow_old: np.ndarray,
                     node: int, comm_src: int, comm_trg: int) -> np.ndarray:
    """Update the community exit flow for a directed graph when one node changes communities.
    This function updates the exit flow incrementally instead of recomputing it from scratch.

    Note: Self-loops are explicitly excluded from both the outgoing and incoming edge sections.

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
    # safety checks
    if communities[node] != comm_src:
        raise ValueError(f"Node {node} is not in source community {comm_src}")
    if comm_src == comm_trg:
        return exit_flow_old.copy()

    exit_flow        = np.array(exit_flow_old, copy=True)
    weights          = np.array(g.es["weight"] if g.is_weighted()
                                else np.ones(g.ecount(), dtype=np.float64))
    out_strength     = np.array(g.strength(mode="out",
                                weights="weight" if g.is_weighted() else None))
    

    # Intentionally includes self-loop weight — keeps flow normalisation correct.
    node_out_strength = out_strength[node]
    node_p            = p[node]

    edges        = np.array(g.get_edgelist(), dtype=int)
    out_edge_ids = np.array(g.incident(node, mode="out"), dtype=int)
    in_edge_ids  = np.array(g.incident(node, mode="in"),  dtype=int)

    # remember, for the exit flow of a community we need consider its outgoing links
    # moving the node to another community affects the exit flows of comm_src and comm_trg 
    # for the other communities the assignment of node doesn't matter because it's external either way
    # so it contributes to the exit flow the same way as before

    # Update exit flow for outgoing edges from the moved node.
    if out_edge_ids.size > 0:
        out_edges = edges[out_edge_ids]
        trg_all   = out_edges[:, 1]
        w_all     = weights[out_edge_ids]

        # KEY FIX: drop the self-loop from the edge list.
        # Without this, the self-loop target carries communities[node] = comm_src,
        # so trg_com != comm_trg is True and it inflates new_exit / exit_flow[comm_trg].
        not_self = trg_all != node
        trg      = trg_all[not_self]
        w_out    = w_all[not_self]

        if trg.size > 0:
            trg_com  = communities[trg]
            # node_out_strength keeps the self-loop weight: correct normalisation.
            flow     = node_p * w_out / node_out_strength

            old_exit = np.sum(flow[trg_com != comm_src])
            new_exit = np.sum(flow[trg_com != comm_trg])
            exit_flow[comm_src] -= old_exit
            exit_flow[comm_trg] += new_exit

    # Update exit flow for incoming edges into the moved node from other nodes.
    # Only sources from comm_src or comm_trg can change whether they are external.
    # For incoming links from other communities it doesn't matter, as they will be external either way
    if in_edge_ids.size > 0:
        in_edges = edges[in_edge_ids]
        src_all  = in_edges[:, 0]
        w_all    = weights[in_edge_ids]

        # KEY FIX: drop the self-loop from the edge list.
        # Without this, src == node has src_com == comm_src, so mask_src is True
        # and the self-loop flow is wrongly added to exit_flow[comm_src].
        not_self = src_all != node
        src      = src_all[not_self]
        w_in     = w_all[not_self]

        if src.size > 0:
            src_com           = communities[src]
            out_strength_safe = np.where(out_strength > 0, out_strength, 1.0)
            flow_in           = p[src] * w_in / out_strength_safe[src]

            # Edges comm_src → node were internal; after the move they exit comm_src.
            mask_src = src_com == comm_src
            if np.any(mask_src):
                exit_flow[comm_src] += np.sum(flow_in[mask_src])

            # Edges comm_trg → node were exiting comm_trg; after the move they are internal.
            mask_trg = src_com == comm_trg
            if np.any(mask_trg):
                exit_flow[comm_trg] -= np.sum(flow_in[mask_trg])

    return exit_flow


def pagerank_nonuniform(M, tau: float = 0.15, tol: float = 1e-15, maxiter: int = 1e6):
    """Two-step PageRank for smart unrecorded teleportation (tutorial Eq. 4-6).
    
    Step 1: solve for p* with teleportation proportional to out-strength.
    Step 2: take one extra link-only step to get the recorded visit rates p.
    
    Nodes with no incoming edges are zeroed out (they can only be reached
    via teleportation, which is unrecorded in this scheme).
    """
    N = M.shape[0]
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    dangling = (row_sums == 0)
    no_incoming = (col_sums == 0)
    row_sums_safe = np.where(dangling, 1, row_sums)
    M_norm = M / row_sums_safe[:, None]

    total_out = row_sums.sum()
    d = row_sums / total_out if total_out > 0 else np.ones(N) / N

    # Step 1
    p_star = np.ones(N) / N
    for _ in range(int(maxiter)):
        dangling_sum = p_star[dangling].sum()
        p_star_new = (1 - tau) * (p_star @ M_norm + dangling_sum * d) + tau * d
        if np.linalg.norm(p_star_new - p_star) < tol:
            p_star = p_star_new
            break
        p_star = p_star_new

    # Step 2: link-only step + dangling redistribution
    dangling_sum = p_star[dangling].sum()
    p = p_star @ M_norm + dangling_sum * d
    p[no_incoming] = 0
    p = p / p.sum()
    return p


def compute_enter_flow_nonuniform(g: ig.Graph, communities: list[int], p: np.ndarray) -> np.ndarray:
    """Rate of flow entering each community via incoming edges from outside."""
    communities = np.array(communities)
    out_strength = np.array(g.strength(mode="out", weights="weight" if g.is_weighted() else None))
    weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))
    edges = np.array(g.get_edgelist(), dtype=int)

    src, trg = edges[:, 0], edges[:, 1]
    src_com, trg_com = communities[src], communities[trg]
    betw = src_com != trg_com

    out_str_safe = np.where(out_strength > 0, out_strength, 1.0)
    flow = p[src] * weights / out_str_safe[src]

    enter_flow = np.zeros(max(communities) + 1)
    np.add.at(enter_flow, trg_com[betw], flow[betw])
    return enter_flow

def update_node_move_description_length(g, communities_old, p_old, p_mod_old, exits_old,
                                         node, comm_trg, tau=0.15,
                                         teleportation="uniform",
                                         returnTerms=False, verbose=False):
    """Compute the change in description length if a single node is moved.

    Args:
        ... (existing args)
        teleportation: "uniform" uses the incremental update (fast).
            "nonuniform" falls back to a full recompute (slower but correct).
    """
    comm_src = communities_old[node]
    if comm_src == comm_trg:
        warnings.warn(f"Node already in target community {comm_trg}! No change.")
        if returnTerms:
            return None, communities_old, p_mod_old, exits_old
        else:
            return None

    # Nonuniform: fall back to full recompute
    if teleportation == "nonuniform":
        communities_old = np.array(communities_old)
        communities_new = communities_old.copy()
        communities_new[node] = comm_trg
        if returnTerms:
            L, p_new, p_mod_new, exit_data = compute_description_length(
                g, communities_new, tau=tau, teleportation="nonuniform",
                returnTerms=True, verbose=verbose
            )
            return L, communities_new, p_mod_new, exit_data
        else:
            return compute_description_length(g, communities_new, tau=tau,
                                              teleportation="nonuniform",
                                              verbose=verbose)

    # === Uniform path: existing incremental update ===
    communities_old = np.array(communities_old)
    communities_new = communities_old.copy()
    communities_new[node] = comm_trg

    num_communities = len(p_mod_old)
    N = g.vcount()

    p_node = p_old[node]
    p_mod_new = p_mod_old.copy()
    p_mod_new[comm_src] -= p_node
    p_mod_new[comm_trg] += p_node

    if g.is_directed():
        node_counts = np.bincount(communities_new, minlength=num_communities)
        exit_flow_new = update_exit_flow(g, communities_old, p_old, exits_old, node, comm_src, comm_trg)
        q_mod = tau * (N - node_counts) / N * p_mod_new + (1 - tau) * exit_flow_new
    else:
        weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))
        total_weight_x2 = 2 * np.sum(weights)
        exit_weights_new = update_exit_weights(g, communities_old, exits_old, node, comm_src, comm_trg)
        q_mod = exit_weights_new / total_weight_x2

    q_sum = np.sum(q_mod)
    p_loop = p_mod_new + q_mod

    if verbose:
        print("p sum:        ", p_old.sum())
        print("p_mod sum:    ", p_mod_new.sum())
        print("q_mod sum:    ", q_mod.sum())

    L = safe_xlogx(q_sum) - 2 * np.sum(safe_xlogx(q_mod)) \
        - np.sum(safe_xlogx(p_old)) + np.sum(safe_xlogx(p_loop))

    exit_data = exit_flow_new if g.is_directed() else exit_weights_new

    if returnTerms:
        return L, communities_new, p_mod_new, exit_data
    else:
        return L

def node_movement_optimization(g, initial_communities=None, teleportation="uniform", returnTerms=False, verbose=False):
    """Phase 1 of the search algorithm. Iteratively moves each node to the
    neighbouring community that minimizes L, until no further improvement.

    Args:
        g: input graph
        initial_communities(list[int], optional): Initial community assignment to start the optimization 
                    from. If None, optimization starts with each node assigned to its own community.   
        teleportation: "uniform" or "nonuniform" (default: uniform).
            For nonuniform, the incremental updates fall back to full recompute,
            making this much slower but still correct.
        returnTerms: whether to also return L, p_mod, exit_data
        verbose: print progress info
    """
    nodes = g.vs.indices
    N_nodes = g.vcount()
    neighborhood = [np.array(nb, dtype=np.intp)
                for nb in g.neighborhood(mindist=1)]

    # if only one node, automatically return it as only community
    # and do not try to optimize
    if N_nodes == 1:
        if verbose:
            print("Graph has only one node, returning it as the only community.")
        if returnTerms:
            L, p, p_mod, exit_data = compute_description_length(
                g, np.array([0]), teleportation=teleportation, returnTerms=True
            )
            return np.array([0]), L, p_mod, exit_data
        else:
            return np.array([0])
        
    # if no initial community assignment is provided,
    # initialize community partition with each node being its own community
    # also do this if there is only 1 community, otherwise there are no neighbouring
    # communities to move to, and the optimization fails
    if initial_communities is None or len(np.unique(initial_communities)) == 1:
        if verbose:
            print("Initialising node movement optimization with each node in its own community.")
        communities = np.arange(N_nodes) # start with each node assigned to its own community
    else: 
        if verbose:
            print("Initialising node movement optimization with the given initial community assignments.")
        communities = initial_communities.copy()
        
    L, p, p_mod, exit_data = compute_description_length(
        g, communities, teleportation=teleportation, returnTerms=True
    )

    if verbose:
        print(f"Starting from description length: {L}")

    optimizable = True
    while optimizable:
        nodes = np.random.permutation(nodes)
        no_move_ctr = 0

        for n in nodes:
            neighbors = neighborhood[n]
            nb_comms = communities[neighbors]
            src_comm = communities[n]
            # set logic supposedly slightly faster than np.unique?
            seen = set()
            comms_to_test = []
            for c in nb_comms:
                ci = int(c)
                if ci != src_comm and ci not in seen:
                    seen.add(ci)
                    comms_to_test.append(ci)

            if not comms_to_test: # if it's empty for some ungodly reason
                no_move_ctr += 1
                continue
            
            # init best params
            L_best = L
            best_comm = src_comm   # to track if moves have been made
            p_mod_best = None
            exit_data_best = None
            communities_best = None
            # go through neighbouring communities:
            for nbc in comms_to_test:
                L_new, communities_new, p_mod_new, exit_data_new = update_node_move_description_length(
                    g, communities, p, p_mod, exit_data, n, nbc,
                    teleportation=teleportation, returnTerms=True
                )
                if L_new is not None and L_new < L_best: # improvement was made
                    L_best = L_new
                    best_comm = nbc
                    p_mod_best = p_mod_new      # already a fresh array from the helper
                    exit_data_best = exit_data_new
                    communities_best = communities_new

            if best_comm == src_comm: # no move has been made
                no_move_ctr += 1 
            else: # update the terms
                L = L_best
                communities = communities_best
                p_mod = p_mod_best
                exit_data = exit_data_best

        # only stop optimizing if not a single improving move has been made in the sequence
        # otherwise keep optimizing
        optimizable = no_move_ctr < N_nodes

        # End-of-sweep: relabel to compact arrays, then reseed L/p/p_mod/exit_data
        # from scratch so all four are consistent with each other before the next sweep
        # (or before returning). This is the single point where relabelling happens.
        _, communities = np.unique(communities, return_inverse=True)
        L, p, p_mod, exit_data = compute_description_length(
            g, communities, teleportation=teleportation, returnTerms=True
        )

        if verbose:
            print(f"Current best description length: {L}")
            print(f"Number of nodes moved this iteration: {N_nodes - no_move_ctr}")
            if optimizable:
                print("Continuing optimization.")
            else: 
                print("Optimization finished!")

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