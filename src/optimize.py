import igraph as ig
import numpy as np
import warnings   
import src.map_equation as meq


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
            L, p, p_mod, exit_data = meq.compute_description_length(
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
        
    L, p, p_mod, exit_data = meq.compute_description_length(
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
                L_new, communities_new, p_mod_new, exit_data_new = meq.update_node_move_description_length(
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
        L, p, p_mod, exit_data = meq.compute_description_length(
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
    

def core_search_algorithm(g:ig.Graph, teleportation="uniform", verbose=False):
    """Runs core algorithm of the infomap community partition search algorithm, without any
    refinement steps. Follows the description in "The map equation" (M. Rosvall, D. Axelsson, and C.T. Bergstrom, 2009).
    Alternates between node-movement optimization and network compression until no further improvements can be made.

    Args:
        g (ig.Graph): Input graph. Supports directed/undirected and weighted/unweighted.
        verbose (bool, optional): Whether to print verbose output for debugging. Defaults to False.

    Returns:
        list[int]: A list of integercommunity labels for each node in the input graph.
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    N = g.vcount() # number of nodes in graph
    flat_comms = np.arange(N, dtype=int) # we'll use this to track how each node in the og graph
    # maps to the supernodes of the current compressed graph

    g_current = g 
    level = 0 # tracking the compression depth

    # start with a loop that repeats as long as we still have improvements:
    while True:
        n_current = g_current.vcount()
        if verbose:
            print(f"\n--- Level {level} ------------------------------")
            print(f"    Current Graph: {n_current} nodes, {g_current.ecount()} edges")

        L_before = meq.compute_description_length(g_current, np.arange(n_current), teleportation=teleportation)
    
        # --- Phase 1: optimization via single-node moves ---
        comms_level, L_after, _, _ = node_movement_optimization(g_current, teleportation=teleportation, returnTerms=True, verbose=verbose)
        n_communities = int(np.max(comms_level)) + 1 # works bc labels should be 0 indexed & contiguous

        if verbose:
            print(f"    Phase 1 found {n_communities} communities")

        # if no improvement has been made in node movement optimization, exit loop
        if L_after >= L_before:
            if verbose:
                print("    Node movement did not improve codelength, stopping optimization.")
            break

        # --- Update node mapping ---
        # get for each node in the current level graph the corresponding community
        # indices after optimization (corresponding to supernode indices of the compressed graph
        # that will be created in the next step
        # this should work because comms_evel is already 0-indexed
        flat_comms = comms_level[flat_comms] 
        
        # --- Phase 2: Network compression ---
        g_current, _ = compress_network(g_current, comms_level, verbose=verbose)
        if verbose: 
            print(f"    Compressed network has description length L = {meq.compute_description_length(g_current, range(n_communities))}")
        level += 1
        
    # Normalise: make community labels contiguous and 0-indexed. 
    # (bc rn the labels could have gaps) 
    # works neatly with the inverse labels from np.unique
    _, flat_comms = np.unique(flat_comms, return_inverse=True)

    # assert that we ended up with proper contiguous labelling
    assert set(flat_comms) == set(range(max(flat_comms)+1)), "Error: non-contiguous or non-0-indexed labels in recursive submodule search result."
    assert len(flat_comms) == g.vcount(), "Error: result length doesn't match number of nodes in graph in recursive submodule search result."
    
    if verbose:
        L_final = meq.compute_description_length(g, flat_comms.tolist(), teleportation=teleportation)
        print(f"\nFinal: {len(np.unique(flat_comms))} communities, "
              f"L = {L_final:.6f} bits")

    return flat_comms.tolist()
        

def search_submodules_with_recursion(g, teleportation="uniform", verbose=False):
    """Recursively split graph into submodules based on its current community assignments,
    and run core algorithm on the subgraphs. To be used in the submodule movement optimization step.

    Args:
        g (igraph.Graph): Input graph. Supports directed/undirected and weighted/unweighted.
        verbose (bool, optional): Whether to print verbose output for debugging. Defaults to False.

    Returns:
        list[int]: A list of integer community labels for each node in the input graph.
    """
    N = g.vcount()
    if N <= 1:
        if verbose:
            print(f"Base case reached with singleton node (N={N}). Stopping recursion.")
        return np.zeros(N, dtype=int)
    
    comms = np.asarray(core_search_algorithm(g, verbose=verbose, teleportation=teleportation), dtype=int)
    unique_comms = np.unique(comms)
    n_comms = len(unique_comms)

    if n_comms == 1:
        if verbose:
            print(f"No improving split for given {N} nodes found. Stopping recursion.")
        return np.zeros(N, dtype=int)
    
    # more efficient than np.where in the loop for building the node-index lists
    order      = np.argsort(comms, kind="stable")
    sorted_c   = comms[order]
    boundaries = np.searchsorted(sorted_c, unique_comms)

    result = np.empty(N, dtype=int)
    next_label = 0

    for i, c in enumerate(unique_comms):
        lo    = boundaries[i]
        hi    = boundaries[i + 1] if i + 1 < n_comms else N
        nodes = order[lo:hi]

        if len(nodes) == 1:
            result[nodes] = next_label
            next_label += 1
            continue

        subgraph = g.induced_subgraph(nodes)
        sub_comms = search_submodules_with_recursion(subgraph, teleportation=teleportation, verbose=verbose)

        for s in np.unique(sub_comms):
            result[nodes[sub_comms == s]] = next_label
            next_label += 1

    # sanity check for proper labelling
    assert set(result) == set(range(next_label)), "Error: non-contiguous or non-0-indexed labels in recursive submodule search result."
    assert len(result) == N, "Error: result length doesn't match number of nodes in graph in recursive submodule search result."

    return result

def submodule_movement_optimization(g: ig.Graph, communities: list[int]=None, teleportation="uniform", verbose=False) -> np.ndarray:
    """Refine a community partition via submodule movements:
    1. Take subgraphs corresponding to each module and run node-movement optimization to get 
        submodules
    2. Compress full network such that supernodes correspond to submodules
    3. Run node-movement optimization again, but startunf from the parent-module assignments

    Args:
        g (ig.Graph): Input graph. Supports directed/undirected and weighted/unweighted.
        communities (list[int] | np.ndarray): Community label for each node. (Labels need
            not be 0-indexed or contiguous).
        verbose (bool, optional): Print progress information. Defaults to False.

    Returns:
        np.ndarray: Refined community assignment for every node of g.
    """
    
    if communities is None:
        if verbose:
            print("No initial partition provided, running core search algorithm first.")
        communities = core_search_algorithm(g, teleportation=teleportation, verbose=verbose)

    communities = np.array(communities, dtype=int)
    L_before = meq.compute_description_length(g, communities, teleportation=teleportation)


    # Normalise community labels to contiguous 0-indexed integers.
    unique_comms = np.unique(communities)          # sorted unique labels
    n_mods       = len(unique_comms)
    comm_idx     = np.searchsorted(unique_comms, communities)   # 0-indexed community per node

    # node indices belonging to each module 
    # this should be faster than the np.where version
    order      = np.argsort(comm_idx, kind="stable")
    sorted_ci  = comm_idx[order]
    boundaries = np.searchsorted(sorted_ci, np.arange(n_mods))
    comm_nodes = []
    for i in range(n_mods):
        lo = boundaries[i]
        hi = boundaries[i + 1] if i + 1 < n_mods else len(order)
        comm_nodes.append(order[lo:hi])

    if verbose:
        print(f"Submodule refinement: Starting with {n_mods} modules, {g.vcount()} nodes and {g.ecount()} edges total.")
        print(f"Initial description length: {L_before:.2f} bits.")

        #for i, (comm, nodes) in enumerate(zip(unique_comms, comm_nodes)):
        #    print(f"  Module {comm} (idx {i}): {len(nodes)} nodes.")

    # Take each module and extract the corresponding subgraph, 
    # run the main node-movement optimisation and map the resulting local labels
    # to globally unique integers, simultaneously recording each submodule's
    # parent module index.
    global_submodule = np.empty(g.vcount(), dtype=int)  # global submodule label per node
    submodule_to_parent: list[int] = []  # parent module idx per submodule

    offset = 0 # offset to help with contiguous submodule labelling 
    for mod_idx, nodes in enumerate(comm_nodes):
        # build subgraph here
        subgraph = g.induced_subgraph(nodes)
        
        if verbose:
            print(f"\n--- Parent module {mod_idx} ---")
            print(f"Nodes: {len(nodes)}")
            print(f"Subgraph: {subgraph.vcount()} nodes, {subgraph.ecount()} edges")
        
        # start recursive process
        local_comms = search_submodules_with_recursion(
                    subgraph, teleportation=teleportation, verbose=verbose
                )
        
        if verbose:
            print(f"local_comms: {local_comms}")
            print(f"unique local submodules: {np.unique(local_comms)}")
            print(f"n_submodules: {len(np.unique(local_comms))}")

        # Normalise local labels to contiguous 0-indexed integers,
        # then shift by offset to make them globally unique.
        local_unique  = np.unique(local_comms) # unique local submodule labels
        local_idx     = np.searchsorted(local_unique, local_comms)  # 0-indexed within module
        n_submodules  = len(local_unique) # number of submodules for the given subgraph

        global_submodule[nodes] = local_idx + offset # global submodule labels

        if verbose:
            print(f"offset: {offset}")
            print(f"local_idx: {local_idx}")
            print(f"global labels assigned: {global_submodule[nodes]}")

        # Each local submodule belongs to the current parent module.
        submodule_to_parent.extend([mod_idx] * n_submodules) # update parent map
        offset += n_submodules # update offset

    submodule_to_parent = np.array(submodule_to_parent, dtype=int)
    n_total_submodules  = offset # total number of submodules

    if verbose:
        print(f"Created {n_total_submodules} submodules across {n_mods} modules.")

    # Network compression: each supernode corresponds to one submodule.
    # also returns unique submodule labels to allow reconstruction of node assignments
    g_compressed, unique_submodule_labels = compress_network(g, global_submodule, verbose=verbose)
    
    # unique_submodule_labels[i] is the global submodule label of compressed node i,
    # so indexing submodule_to_parent with it gives the parent module index for each compressed node.
    initial_compressed_comms = submodule_to_parent[unique_submodule_labels]

    if verbose:
        print(f"  Compressed nodes: {g_compressed.vcount()}, "
              f"unique initial communities: {len(np.unique(initial_compressed_comms))}")

    # Performing node-movement optimization on compressed network, with an initial
    # community partition corresponding to the result of the core search algorithm
    # so each compressed node (submodule) is initially assigned to its parent module.
    final_compressed_comms = np.array(
        node_movement_optimization(g_compressed,
                                       initial_communities=initial_compressed_comms,
                                       teleportation=teleportation,
                                       verbose=verbose,
                                       ),
        dtype=int,
    )

    if verbose:
        print("\nAfter compressed optimisation:")
        print(f"final_compressed_comms: {final_compressed_comms}")

    # map the refined partition back to og nodes
    compressed_node_idx = np.searchsorted(unique_submodule_labels, global_submodule)
    final_communities   = final_compressed_comms[compressed_node_idx]
    L_after  = meq.compute_description_length(g, final_communities, teleportation=teleportation)

    if verbose:
        n_final = len(np.unique(final_communities))
        print(f"Submodule refinement complete: {n_mods} -> {n_final} communities.")
        print(f"Description length: {L_before} -> {L_after}")

        # assert that we ended up with proper contiguous labelling
    assert set(final_communities) == set(range(max(final_communities)+1)), "Error: non-contiguous or non-0-indexed labels in recursive submodule search result."
    assert len(final_communities) == g.vcount(), "Error: result length doesn't match number of nodes in graph in recursive submodule search result."

    # one last check to be super sure to only accept improvments
    if L_after < L_before:
        return final_communities
    else:
        return communities
    

    
def search_community_partition(g:ig.Graph, num_restarts:int=10, max_iter:int=100, teleportation="uniform", verbose:bool=False):
    """Find community partition for a given network via a greedy search algorithm minimizing the description length.
    Performs num_restart independent searches and returns the partition with the lowest description length. Optimization consists of
    1. Running the greedy core search algorithm
    2. Iterating submodule movement refinement and single-node movement refinement of the core result until no improvements can be
       achieved or max_iter iterations have been reached.


    Args:
        g (ig.Graph): Input graph. Supports directed/undirected and weighted/unweighted.
        num_restarts (int, optional): Number of independent restarts of the search process. Defaults to 10.
        max_iter (int, optional): Maximum number of refinement iterations. Defaults to 100.
        teleportation (str, optional): Which teleportation scheme to use for the Infomap description length computation. Must be "uniform" or "nonuniform". Defaults to "uniform".
        verbose (bool, optional): Whether to print verbose output for debugging. Defaults to False.

    Returns:
        list[int]: Best found community partition, as a list of community labels for each node.
    """

    N = g.vcount() # number of nodes in graph
    if verbose:
        print(f"--- Running community search ---------------------------------------")
        print(f"Input Graph: {N} nodes, {g.ecount()} edges")

    comms_best = np.empty(N)
    L_best = np.inf

    for j in range(num_restarts): # number of independent runs
        if verbose:
            print(f"\n--- Run {j+1}/{num_restarts}:  Starting community partition search... -------")
            print(f"Starting from description length L = {meq.compute_description_length(g, np.arange(N), teleportation=teleportation)} bits (with trivial parititon)")

        comms_initial = core_search_algorithm(g, teleportation=teleportation, verbose=verbose) # runs core optimization

        if verbose:
            L_initial = meq.compute_description_length(g, comms_initial, teleportation=teleportation)
            print(f"Initial partition found by core search algorithm has description length L = {L_initial:.6f} bits")
            print(f"--- Starting refinement process...\n")

        # start with a loop that repeats as long as we still have improvements:
        comms_level = comms_initial
        for i in range(max_iter):
            if verbose:
                print(f"\n--- Refinement: Starting Iteration {i+1}")
            L_before = meq.compute_description_length(g, comms_level, teleportation=teleportation)
            # submodule refinement
            comms_level = submodule_movement_optimization(g, comms_level, teleportation=teleportation, verbose=verbose)
            # single-node refinements
            comms_level = node_movement_optimization(g, comms_level, teleportation=teleportation, verbose=verbose)
            L_after = meq.compute_description_length(g, comms_level, teleportation=teleportation)
            if verbose:
                print(f"--- Refinement: Finished Iteration {i+1}")
                print(f"        L_before={L_before:.6f}, L_after={L_after:.6f}")
                print(f"        Current number of communities: {len(np.unique(comms_level))}")
            if (L_after >= L_before):
                break
            
        if i >= max_iter-1:
            warnings.warn(f"Reached maximum number of iterations ({max_iter}), stopping refinement process.")


        comms_unique = np.unique(comms_level) # get sorted list of unique communities
        n_communities = len(comms_unique) # number of unique communities

        if verbose:
            print(f"\n Run {j+1}: Found {n_communities} communities, "
                f"L = {L_after:.6f} bits")
            
        if L_best > L_after: # if this is the best description length so far:
            L_best = L_after
            comms_best = comms_level.copy()

    return comms_best.tolist()
