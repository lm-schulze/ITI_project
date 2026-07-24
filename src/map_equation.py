# library imports
import igraph as ig
import numpy as np
import warnings
import scipy.sparse as sp
import numba


def build_sparse_adjacency(g: ig.Graph, edges=None, weights=None) -> sp.csr_matrix:
    """Build a sparse (CSR) (weighted) adjacency matrix for a (directed) graph.

    This replaces g.get_adjacency(), in order to avoid memory issues when 
    working with large graphs.

    Args:
        g (ig.Graph): Input graph.

    Returns:
        sp.csr_matrix: Sparse adjacency matrix.
    """
    N = g.vcount()
    if g.ecount() == 0:
        return sp.csr_matrix((N, N))

    if edges is None:
        edges = np.array(g.get_edgelist(), dtype=np.int64)

    if weights is None:
        weights = np.array(
            g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64)
            )
        
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()
    return adj

# compute x*log2(x) and safely handle log(0) issues:
# by safely handle I mean just set it to zero
# split up the scalar and array case for numba
@numba.njit(cache=True)
def _safe_xlogx_scalar(x):
    if x > 0.0:
        return x * np.log2(x)
    return 0.0


@numba.njit(cache=True)
def _safe_xlogx_arr(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]): # we're working with numba so the for loop is actually fine :D
        xi = x[i]
        if xi > 0.0:
            out[i] = xi * np.log2(xi)
        else:
            out[i] = 0.0
    return out


# compute x*log2(x) and safely handle log(0) issues:
# by safely handle I mean just set it to zero
# now just a wrapper for the numba versions :D
def safe_xlogx(x):
    """Compute x*log2(x) safely, setting log(0) to zero.

    Args:
        x: input value or array for which to compute x*log2(x)

    Returns:
        x*log2(x) for x > 0, and 0 for x <= 0
    """
    if np.isscalar(x):
        return _safe_xlogx_scalar(float(x))
    x = np.asarray(x, dtype=np.float64)
    return _safe_xlogx_arr(x)


def compute_exit_weights(g: ig.Graph, communities: list[int], weights=None, edges=None) -> np.ndarray:
    """Compute community exit weights for a given undirected graph and community partition.
       Helper function for the description length computation via map equation.

    Args:
        g (ig.Graph): (Undirected) input graph.
        communities (list[int]): List of non-overlapping community labels for all nodes of input graph G.

    Returns:
        np.ndarray: Exit weights for each community.
    """
    if weights is None:
        weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))

    if edges is None:
        edges = np.array(g.get_edgelist(), dtype=int) # array of edges

    communities = np.asarray(communities) # community membership list for each node
    exit_weights = np.zeros(max(communities) + 1) # initialise exit weight array

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
def compute_exit_flow(g: ig.Graph, communities: list[int], p: np.ndarray, weights=None, edges=None, out_strength=None) -> np.ndarray:
    """Compute community exit flow for a given directed graph, community partition, and node visit frequencies.
       Helper function for the description length computation via map equation.

    Args:
        g (ig.Graph): (Directed) input graph.
        communities (list[int]): List of non-overlapping community labels for all nodes of input graph G.
        p (np.ndarray): Node visit frequencies.

    Returns:
        np.ndarray: Exit flow for each community.
    """

    communities = np.asarray(communities) # community membership list for each node

    if out_strength is None:
        out_strength = np.array(g.strength(mode="out", weights="weight" if g.is_weighted() else None)) # strength of outgoing links for each node
    if weights is None:
        weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))
    if edges is None:
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


# originally based off of the PageRank Wikipedia, hehe
# but changed to row-stochastic, and with dangling node handling
def pagerank(M, tau: float = 0.15, tol: float = 1e-10, maxiter: int = 1e6):
    """PageRank algorithm with teleportation probability tau. Returns ranking of nodes (pages) in the adjacency matrix.

    Parameters
    ----------
    M : scipy.sparse matrix or numpy array
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
    is_sparse = sp.issparse(M) # if we're working with a sparse matrix

    row_sums = np.asarray(M.sum(axis=1)).ravel() # corrsponds to out strength
    dangling = (row_sums == 0) # dangling nodes (no outgoing edges)
    row_sums_safe = np.where(dangling, 1, row_sums) # set to one for normalisation

    if is_sparse:
        M_normalised = sp.diags((1.0 / row_sums_safe)) @ M   # row-stochastic: T[i,j] = p(i->j)
        M_normalised = M_normalised.tocsr()
    else:
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


def pagerank_nonuniform(M, tau: float = 0.15, tol: float = 1e-15, maxiter: int = 1e6):
    """Two-step PageRank for smart unrecorded teleportation (tutorial Eq. 4-6).
    
    Step 1: solve for p* with teleportation proportional to out-strength.
    Step 2: take one extra link-only step to get the recorded visit rates p.
    
    Nodes with no incoming edges are zeroed out (they can only be reached
    via teleportation, which is unrecorded in this scheme).
    """
    N = M.shape[0]
    is_sparse = sp.issparse(M)

    # .sum() on a sparse matrix returns a np.matrix -> flatten to a plain 1D array
    row_sums = np.asarray(M.sum(axis=1)).ravel()
    col_sums = np.asarray(M.sum(axis=0)).ravel()
    dangling = (row_sums == 0)
    no_incoming = (col_sums == 0)
    row_sums_safe = np.where(dangling, 1, row_sums)

    if is_sparse:
        # Sparse row-rescaling: keeps nnz(M_norm) == nnz(M), no densification.
        M_norm = (sp.diags(1.0 / row_sums_safe) @ M).tocsr()
    else:
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


def compute_enter_flow_nonuniform(g: ig.Graph,
                                  communities: list[int],
                                  p: np.ndarray,
                                  edges=None,
                                  weights=None,
                                  out_strength=None
                                  ) -> np.ndarray:
    """
        Rate of flow entering each community via incoming edges from outside.
    """
    communities = np.asarray(communities)
    if out_strength is None:
        out_strength = np.array(g.strength(mode="out", weights="weight" if g.is_weighted() else None))
    if weights is None:
        weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))
    if edges is None:
        edges = np.array(g.get_edgelist(), dtype=int)

    src, trg = edges[:, 0], edges[:, 1]
    src_com, trg_com = communities[src], communities[trg]
    betw = src_com != trg_com

    out_str_safe = np.where(out_strength > 0, out_strength, 1.0)
    flow = p[src] * weights / out_str_safe[src]

    enter_flow = np.zeros(max(communities) + 1)
    np.add.at(enter_flow, trg_com[betw], flow[betw])
    return enter_flow


def compute_description_length(g, communities,
                               edges=None,
                               weights=None, 
                               out_strength=None, 
                               adj=None,
                               p=None,
                               sum_xlogx_p=None,
                               total_weight_x2=None,
                               tau=0.15, 
                               teleportation="uniform",
                               returnTerms=False,
                               verbose=False):
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
    
    communities = np.asarray(communities)

    # relabel just in case for 0-indexed, contiguous labels
    _, communities = np.unique(communities, return_inverse=True)
    num_communities = int(communities.max()) + 1 
    N = g.vcount()

    # handle the edge-case (hehe) of a graph without edges, or nodes:
    if g.ecount() == 0 or N == 0: # graph doesn't have edges or nodes:
        if returnTerms:
            empty_scalars = {"q_sum": 0.0, "sum_xlogx_q": 0.0,
                             "sum_xlogx_ploop": 0.0, "sum_xlogx_p": 0.0}
            return 0.0, np.zeros(N), np.zeros(num_communities), np.zeros(num_communities), empty_scalars
        else:   
            return 0.0

    if g.is_directed():
        #adj = np.array(g.get_adjacency(attribute="weight" if g.is_weighted() else None).data, dtype=float)
        if adj is None:
            adj = build_sparse_adjacency(g, edges=edges, weights=weights) # for handling large graphs

        if teleportation == "uniform":
            # === Uniform recorded teleportation ===
            if p is None:
                p = pagerank(adj, tau=tau)
            p_mod = np.zeros(num_communities)
            np.add.at(p_mod, communities, p)
            exit_flow = compute_exit_flow(g, communities, p,
                                          edges=edges,
                                          weights=weights,
                                          out_strength=out_strength)

            # q_mod includes the teleportation term
            n_mod = np.bincount(communities, minlength=num_communities)
            q_mod = tau * (N - n_mod) / N * p_mod + (1 - tau) * exit_flow
            
            # assemble terms - symmetric formula (same qs)
            if sum_xlogx_p is None:
                sum_xlogx_p = np.sum(safe_xlogx(p))

            q_sum = np.sum(q_mod)
            p_loop = p_mod + q_mod
            sum_xlogx_q = np.sum(safe_xlogx(q_mod))
            sum_xlogx_ploop = np.sum(safe_xlogx(p_loop))
            
            L = safe_xlogx(q_sum) - 2 * sum_xlogx_q - sum_xlogx_p + sum_xlogx_ploop
            exit_data = exit_flow

            
        else:  # nonuniform
            # === Smart unrecorded teleportation ===
            if p is None:
                p = pagerank_nonuniform(adj, tau=tau)
            p_mod = np.zeros(num_communities)
            np.add.at(p_mod, communities, p)
            exit_flow = compute_exit_flow(g, communities, p,
                                          edges=edges,
                                          weights=weights,
                                          out_strength=out_strength)
            enter_flow = compute_enter_flow_nonuniform(g, communities, p,
                                                       edges=edges,
                                                       weights=weights,
                                                       out_strength=out_strength)
            
            # asymmetric formula: enter for index, exit for module
            # assemble terms
            if sum_xlogx_p is None:
                sum_xlogx_p = np.sum(safe_xlogx(p))
            
            q_enter = enter_flow
            q_exit = exit_flow
            q_enter_sum = np.sum(q_enter)
            p_loop = p_mod + q_exit
            sum_xlogx_ploop = np.sum(safe_xlogx(p_loop))

            L = safe_xlogx(q_enter_sum) - np.sum(safe_xlogx(q_enter)) \
                - np.sum(safe_xlogx(q_exit)) \
                - sum_xlogx_p + sum_xlogx_ploop
            exit_data = exit_flow
            # for returnTerms/verbose compatibility:
            # (even though we don't need these in the nonuniform teleportation case
            # since we didn't implement any update functions and just recompute)
            q_mod = exit_flow   
            q_sum = q_enter_sum
            sum_xlogx_q = np.sum(safe_xlogx(q_exit))

    else:
        # === Undirected case - same for both teleportation schemes ===
        if weights is None:
            weights = np.array(g.es["weight"] if g.is_weighted() else np.ones(g.ecount(), dtype=np.float64))

        if total_weight_x2 is None:
            total_weight_x2 = 2 * np.sum(weights)

        if p is None:
            p = np.array(g.strength(weights="weight" if g.is_weighted() else None)) / total_weight_x2

        p_mod = np.zeros(num_communities)
        np.add.at(p_mod, communities, p)
        
        exit_weights = compute_exit_weights(g, communities, edges=edges, weights=weights)
        q_mod = exit_weights / total_weight_x2

        # assemble terms
        if sum_xlogx_p is None:
            sum_xlogx_p = np.sum(safe_xlogx(p))

        q_sum = np.sum(q_mod)
        p_loop = p_mod + q_mod
        sum_xlogx_q = np.sum(safe_xlogx(q_mod))
        sum_xlogx_ploop = np.sum(safe_xlogx(p_loop))

        L = safe_xlogx(q_sum) - 2 * sum_xlogx_q - sum_xlogx_p + sum_xlogx_ploop
        exit_data = exit_weights
    
    if verbose:
        print(f"teleportation: {teleportation}")
        print("p sum:        ", p.sum())
        print("p_mod sum:    ", p_mod.sum())
        print("exit_data sum:", exit_data.sum())
        print("q_mod sum:    ", q_mod.sum())
        print("p_loop sum:   ", p_loop.sum())

    if returnTerms:
        scalars = {
            "q_sum": q_sum,
            "sum_xlogx_q": sum_xlogx_q,
            "sum_xlogx_ploop": sum_xlogx_ploop,
            "sum_xlogx_p": sum_xlogx_p,
        }
        return L, p, p_mod, exit_data, scalars

    else:
        return L
    
#######################################################################
## UPDATE FUNCTIONS FOR SINGLE-NODE MOVEMENTS #########################
#######################################################################

# updating exit weights:
# seperate the core part of the update exit weights mechanism that 
# numba can work with from the part working with types numba cannot handle
# i.e. all the igraph stuff, or the incidence dict

@numba.njit(cache=True)
def _update_exit_weights_core(communities, exit_weights_old, node, node_idx, comm_src, comm_trg,
                              edges, weights, out_eids, out_idxptr):
    """Numba-jitted arithmetic core for update_exit_weights.

    Loops directly over the node's incident edges via the CSR-style
    incidence index (out_eids/out_idxptr). Self-loops (neighbor == node)
    are skipped inline..
    """
    start = out_idxptr[node_idx]
    end = out_idxptr[node_idx + 1]

    total_degree = 0.0
    W_src = 0.0
    W_trg = 0.0

    for k in range(start, end):
        eid = out_eids[k]
        a = edges[eid, 0]
        b = edges[eid, 1]
        neighbor = b if a == node else a
        if neighbor == node:  # self-loop: never crosses a community boundary, skip
            continue
        w = weights[eid]
        total_degree += w
        nc = communities[neighbor]
        if nc == comm_src:
            W_src += w
        elif nc == comm_trg:
            W_trg += w

    delta_src = 2.0 * W_src - total_degree
    delta_trg = total_degree - 2.0 * W_trg

    exit_weights_new = exit_weights_old.copy()
    exit_weights_new[comm_src] += delta_src
    exit_weights_new[comm_trg] += delta_trg
    return exit_weights_new

    
def update_exit_weights(g: ig.Graph, 
                        communities_old: list[int], 
                        exit_weights_old: np.ndarray,
                        node: int, 
                        comm_src: int, 
                        comm_trg: int,
                        edges=None,
                        weights=None,
                        incidence_dict=None
                        ) -> np.ndarray:
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
    communities = np.asarray(communities_old)
    # safety checks
    if communities[node] != comm_src:
        raise ValueError(f"Node {node} is not in source community {comm_src}")
    if comm_src == comm_trg:
        return exit_weights_old.copy()

    if incidence_dict is None: # this shouldn't happen in the optimization, ideally
        incident_eids = np.array(g.incident(node), dtype=int)
        ip = np.array([0, len(incident_eids)], dtype=np.int64)
        node_idx = 0
    else:
        ip = incidence_dict["out_idxptr"]  # for undirected, out == in == all-incident
        incident_eids = incidence_dict["out_eids"]
        node_idx = node

    if edges is None:
        edges = np.array(g.get_edgelist(), dtype=int)
    if weights is None:
        weights = np.array(g.es["weight"] if g.is_weighted() 
                               else np.ones(g.ecount(), dtype=np.float64)
                               )
    # calls the numba-adjusted core function
    return _update_exit_weights_core(
            communities.astype(np.int64), exit_weights_old, node, node_idx,
            comm_src, comm_trg, edges, weights, incident_eids, ip
        )

# analogously for exit flows:
@numba.njit(cache=True)
def _update_exit_flow_core(communities, p, exit_flow_old, node, node_idx, comm_src, comm_trg,
                           edges, weights, out_strength,
                           out_eids, out_idxptr, in_eids, in_idxptr):
    """Numba-jitted arithmetic core for update_exit_flow.
    """
    exit_flow = exit_flow_old.copy()

    node_out_strength = out_strength[node]  # includes self-loop weight, as in the original
    #node_out_strength_safe = node_out_strength if node_out_strength > 0.0 else 1.0
    node_p = p[node]

    # --- outgoing edges from node ---
    # remember, for the exit flow of a community we need consider its outgoing links
    # moving the node to another community affects the exit flows of comm_src and comm_trg 
    # for the other communities the assignment of node doesn't matter because it's external 
    # either way, so it contributes to the exit flow the same way as before

    out_start = out_idxptr[node_idx]
    out_end = out_idxptr[node_idx + 1]

    old_exit = 0.0
    new_exit = 0.0
    for k in range(out_start, out_end):
        eid = out_eids[k]
        trg = edges[eid, 1]
        if trg == node:  # self-loop, never crosses a community boundary
            continue
        trg_com = communities[trg]
        w = weights[eid]
        flow = node_p * w / node_out_strength # no clue how a divide by 0 could have happened here, but it did :((
        if trg_com != comm_src:
            old_exit += flow
        if trg_com != comm_trg:
            new_exit += flow

    exit_flow[comm_src] -= old_exit
    exit_flow[comm_trg] += new_exit

    # --- incoming edges into node ---
    # Update exit flow for incoming edges into the moved node from other nodes.
    # Only sources from comm_src or comm_trg can change whether they are external.
    # For incoming links from other communities it doesn't matter, as they will be external either way
    in_start = in_idxptr[node_idx]
    in_end = in_idxptr[node_idx + 1]

    add_to_src = 0.0
    sub_from_trg = 0.0
    for k in range(in_start, in_end):
        eid = in_eids[k]
        src = edges[eid, 0]
        if src == node:  # self-loop, skip
            continue
        w = weights[eid]
        src_out_strength = out_strength[src]
        src_out_strength_safe = src_out_strength if src_out_strength > 0.0 else 1.0
        flow_in = p[src] * w / src_out_strength_safe
        src_com = communities[src]
        if src_com == comm_src:
            add_to_src += flow_in
        elif src_com == comm_trg:
            sub_from_trg += flow_in

    exit_flow[comm_src] += add_to_src
    exit_flow[comm_trg] -= sub_from_trg

    return exit_flow


def update_exit_flow(g: ig.Graph, 
                     communities_old: list[int], 
                     p: np.ndarray,
                     exit_flow_old: np.ndarray,
                     node: int, 
                     comm_src: int, 
                     comm_trg: int,
                     edges=None,
                     weights=None,
                     out_strength=None,
                     incidence_dict=None
                     ) -> np.ndarray:
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
    communities = np.asarray(communities_old)
    # safety checks
    if communities[node] != comm_src:
        raise ValueError(f"Node {node} is not in source community {comm_src}")
    if comm_src == comm_trg:
        return exit_flow_old.copy()

    if weights is None:
        weights = np.array(g.es["weight"] if g.is_weighted()
                           else np.ones(g.ecount(), dtype=np.float64))
    if out_strength is None:
        out_strength = np.array(g.strength(mode="out", weights="weight"
                                           if g.is_weighted() else None))
    if edges is None:
        edges = np.array(g.get_edgelist(), dtype=int)

    if incidence_dict is None:
        out_eids = np.array(g.incident(node, mode="out"), dtype=np.int64)
        in_eids  = np.array(g.incident(node, mode="in"),  dtype=np.int64)
        out_idxptr = np.array([0, len(out_eids)], dtype=np.int64)
        in_idxptr = np.array([0, len(in_eids)], dtype=np.int64)
        node_idx = 0
    else:
        out_idxptr = incidence_dict["out_idxptr"]
        out_eids = incidence_dict["out_eids"]
        in_idxptr = incidence_dict["in_idxptr"]
        in_eids = incidence_dict["in_eids"]
        node_idx = node

    return _update_exit_flow_core(
        communities.astype(np.int64), p, exit_flow_old, node, node_idx,
        comm_src, comm_trg, edges, weights, out_strength,
        out_eids, out_idxptr, in_eids, in_idxptr
    )


def update_node_move_description_length_old(g,  
                                        communities_old,
                                        p_old, 
                                        p_mod_old, 
                                        exits_old,
                                        node, 
                                        comm_trg,
                                        node_counts_old=None, 
                                        edges=None,
                                        weights=None,
                                        out_strength=None,
                                        incidence_dict=None,
                                        total_weight_x2=None,
                                        tau=0.15,
                                        teleportation="uniform",
                                        returnTerms=False,
                                        verbose=False
                                        ):
    """Update the description length if a single node is moved.

    Args:
        ... (existing args)
        teleportation: "uniform" uses the incremental update (fast).
            "nonuniform" falls back to a full recompute (slower but correct).
    """
    comm_src = communities_old[node]
    if comm_src == comm_trg:
        warnings.warn(f"Node already in target community {comm_trg}! No change.")
        if returnTerms:
            return None, node_counts_old, p_mod_old, exits_old
        else:
            return None

    # Nonuniform: fall back to full recompute
    # TODO: implement nonuniform update funcs, if we have the time
    if teleportation == "nonuniform":
        communities_old = np.asarray(communities_old)
        communities_new = communities_old.copy()
        communities_new[node] = comm_trg
        if returnTerms:
            L, _, p_mod_new, exit_data = compute_description_length(
                g, communities_new, tau=tau, teleportation="nonuniform",
                edges=edges, weights=weights, out_strength=out_strength,
                returnTerms=True, verbose=verbose
            )
            return L, communities_new, p_mod_new, exit_data
        else:
            return compute_description_length(g, communities_new, tau=tau,
                                              teleportation="nonuniform",
                                              edges=edges, 
                                              weights=weights,
                                              out_strength=out_strength,
                                              verbose=verbose)

    # === Uniform path: existing incremental update ===
    # communities_old = np.asarray(communities_old)
    # communities_new = communities_old.copy()
    # communities_new[node] = comm_trg

    num_communities = len(p_mod_old)
    N = g.vcount()

    p_node = p_old[node]
    p_mod_new = p_mod_old.copy()
    p_mod_new[comm_src] -= p_node
    p_mod_new[comm_trg] += p_node

    if g.is_directed():
        if node_counts_old is None: # THIS SHOULDN'T HAPPEN!! (but just in case)
            communities_new = communities_old.copy()
            communities_new[node] = comm_trg
            node_counts = np.bincount(communities_new, minlength=num_communities)
        else:
            # only comm_src and comm_trg change, by exactly 1 node each.
            node_counts = node_counts_old.copy()   # O(num_communities), not O(N)
            node_counts[comm_src] -= 1
            node_counts[comm_trg] += 1

        exit_flow_new = update_exit_flow(g, communities_old,
                                         p_old, exits_old,
                                         node, comm_src, comm_trg,
                                         edges=edges, weights=weights, 
                                         out_strength=out_strength,
                                         incidence_dict=incidence_dict)
        q_mod = tau * (N - node_counts) / N * p_mod_new + (1 - tau) * exit_flow_new
    else:
        if weights is None:
            weights = np.array(g.es["weight"] if g.is_weighted()
                               else np.ones(g.ecount(), dtype=np.float64))

        if total_weight_x2 is None:
            total_weight_x2 = 2 * np.sum(weights)
        exit_weights_new = update_exit_weights(g, communities_old, exits_old, node,
                                               comm_src, comm_trg,
                                               edges=edges, weights=weights,
                                               incidence_dict=incidence_dict)
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
    node_counts_new = node_counts if g.is_directed() else None  # urgh this is ugly but I'm not sure how to handle it better 

    if returnTerms:
        return L, node_counts_new, p_mod_new, exit_data
    else:
        return L


def update_node_move_description_length(g,
                                        communities_old,
                                        p_old,
                                        p_mod_old,
                                        exits_old,
                                        node,
                                        comm_trg,
                                        scalars_old=None,
                                        node_counts_old=None,
                                        edges=None,
                                        weights=None,
                                        out_strength=None,
                                        incidence_dict=None,
                                        total_weight_x2=None,
                                        tau=0.15,
                                        teleportation="uniform",
                                        returnTerms=False,
                                        verbose=False
                                        ):

    """Compute the difference in description length if a single node is moved.

    Args:
        ... (existing args)
        teleportation: "uniform" uses the incremental update (fast).
            "nonuniform" falls back to a full recompute (slower but correct).
    """
    comm_src = communities_old[node]
    if comm_src == comm_trg:
        warnings.warn(f"Node already in target community {comm_trg}! No change.")
        if returnTerms:
            return None, node_counts_old, p_mod_old, exits_old
        else:
            return None

    # Nonuniform: fall back to full recompute
    # TODO: implement nonuniform update funcs, if we have the time
    if teleportation == "nonuniform":
        communities_old = np.asarray(communities_old)
        communities_new = communities_old.copy()
        communities_new[node] = comm_trg
        if returnTerms:
            L, _, p_mod_new, exit_data, scalars_new = compute_description_length(
                g, communities_new, tau=tau, teleportation="nonuniform",
                edges=edges, weights=weights, out_strength=out_strength,
                returnTerms=True, verbose=verbose
            )
            return L, communities_new, p_mod_new, exit_data, scalars_new

        else:
            return compute_description_length(g, communities_new, tau=tau,
                                              teleportation="nonuniform",
                                              edges=edges, 
                                              weights=weights,
                                              out_strength=out_strength,
                                              verbose=verbose)

    if scalars_old is None:
        raise ValueError("scalars_old is required for the efficient description length update."
                        "Obtain it from compute_description_length(..., returnTerms=True) "
                        "or a prior call to this function.")

    # === Uniform path: existing incremental update ===
    # so technically, q_mod, p_mod (and therefore p_loop) only change in 2 indices,
    # comm_src and comm_trg, and so we should be able to just compute those diffs
    # instead of working with the full arrays of length num_communities

    num_communities = len(p_mod_old)
    N = g.vcount()

    p_node = p_old[node]
    p_mod_new = p_mod_old.copy()
    p_mod_new[comm_src] -= p_node
    p_mod_new[comm_trg] += p_node

    if g.is_directed():
        if node_counts_old is None: # THIS SHOULDN'T HAPPEN!! (but just in case)
            communities_new = communities_old.copy()
            communities_new[node] = comm_trg
            node_counts = np.bincount(communities_new, minlength=num_communities)
        else:
            # only comm_src and comm_trg change, by exactly 1 node each.
            node_counts = node_counts_old.copy()   # O(num_communities), not O(N)
            node_counts[comm_src] -= 1
            node_counts[comm_trg] += 1

        exit_flow_new = update_exit_flow(g, communities_old,
                                         p_old, exits_old,
                                         node, comm_src, comm_trg,
                                         edges=edges, weights=weights, 
                                         out_strength=out_strength,
                                         incidence_dict=incidence_dict)
        # compute the new q_mods for the src/trg comms only
        q_src_new = (tau * (N - node_counts[comm_src]) / N * p_mod_new[comm_src]
                    + (1 - tau) * exit_flow_new[comm_src])
        q_trg_new = (tau * (N - node_counts[comm_trg]) / N * p_mod_new[comm_trg]
                    + (1 - tau) * exit_flow_new[comm_trg])
        
        exit_data_new = exit_flow_new

    else:
        if weights is None:
            weights = np.array(g.es["weight"] if g.is_weighted()
                               else np.ones(g.ecount(), dtype=np.float64))

        if total_weight_x2 is None:
            total_weight_x2 = 2 * np.sum(weights)

        exit_weights_new = update_exit_weights(g, communities_old, exits_old, node,
                                               comm_src, comm_trg,
                                               edges=edges, weights=weights,
                                               incidence_dict=incidence_dict)
        q_src_new = exit_weights_new[comm_src] / total_weight_x2
        q_trg_new = exit_weights_new[comm_trg] / total_weight_x2
        exit_data_new = exit_weights_new
        node_counts = None

    # get the old q_mod at src comm
    q_src_old = ( (tau * (N - (node_counts_old[comm_src] if node_counts_old is not None else 0)) / N
                  * p_mod_old[comm_src] + (1 - tau) * exits_old[comm_src])
                 if g.is_directed() else exits_old[comm_src] / total_weight_x2 )
    # get the old q_modat trg comm
    q_trg_old = ( (tau * (N - (node_counts_old[comm_trg] if node_counts_old is not None else 0)) / N
                  * p_mod_old[comm_trg] + (1 - tau) * exits_old[comm_trg])
                 if g.is_directed() else exits_old[comm_trg] / total_weight_x2 )

    # get the old/new p_loop terms at comm_src/trg
    pl_src_old = p_mod_old[comm_src] + q_src_old
    pl_trg_old = p_mod_old[comm_trg] + q_trg_old
    pl_src_new = p_mod_new[comm_src] + q_src_new
    pl_trg_new = p_mod_new[comm_trg] + q_trg_new

    # now update the running scalars (q_sum, sum_xlogx_q, sum_xlogx_ploop)
    q_sum_new = (scalars_old["q_sum"] - q_src_old - q_trg_old \
                 + q_src_new + q_trg_new)
 
    sum_xlogx_q_new = (scalars_old["sum_xlogx_q"]
                      - safe_xlogx(q_src_old) - safe_xlogx(q_trg_old)
                      + safe_xlogx(q_src_new) + safe_xlogx(q_trg_new))
 
    sum_xlogx_ploop_new = (scalars_old["sum_xlogx_ploop"]
                          - safe_xlogx(pl_src_old) - safe_xlogx(pl_trg_old)
                          + safe_xlogx(pl_src_new) + safe_xlogx(pl_trg_new))

    # this one remains constant
    sum_xlogx_p = scalars_old["sum_xlogx_p"]  

    # compute description length
    L = safe_xlogx(q_sum_new) - 2 * sum_xlogx_q_new - sum_xlogx_p + sum_xlogx_ploop_new

    # update running scalar dict with new values
    scalars_new = {
        "q_sum": q_sum_new,
        "sum_xlogx_q": sum_xlogx_q_new,
        "sum_xlogx_ploop": sum_xlogx_ploop_new,
        "sum_xlogx_p": sum_xlogx_p,
    }

    if verbose:
        print("p sum:        ", p_old.sum())
        print("p_mod sum:    ", p_mod_new.sum())
        print("q_mod sum:    ", q_sum_new)

    node_counts_new = node_counts if g.is_directed() else None
 
    if returnTerms:
        return L, node_counts_new, p_mod_new, exit_data_new, scalars_new
    else:
        return L