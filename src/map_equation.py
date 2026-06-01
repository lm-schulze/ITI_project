# map_equation.py
# Map equation implementation: uniform and smart unrecorded teleportation schemes

import igraph as ig
import numpy as np
import warnings


def safe_xlogx(x):
    """Compute x*log2(x), returning 0 when x <= 0."""
    return np.where(x > 0, x * np.log2(x), 0.0)


# =======================
# Uniform teleportation
# =======================

def pagerank_uniform(M, tau=0.15, tol=1e-15, maxiter=int(1e6)):
    """Standard PageRank with uniform teleportation (d = 1/N for all nodes)."""
    N = M.shape[0]
    row_sums = M.sum(axis=1)
    dangling = (row_sums == 0)
    row_sums_safe = np.where(dangling, 1, row_sums)
    M_norm = M / row_sums_safe[:, None]

    d = np.ones(N) / N

    p = np.ones(N) / N
    for _ in range(int(maxiter)):
        dangling_sum = p[dangling].sum()
        p_new = (1 - tau) * (p @ M_norm + dangling_sum * d) + tau * d
        if np.linalg.norm(p_new - p) < tol:
            return p_new
        p = p_new
    warnings.warn(f"pagerank_uniform did not converge after {maxiter} iterations.")
    return p


def compute_description_length_uniform(g, communities, tau=0.15):
    """Map equation L using uniform recorded teleportation."""
    communities = np.array(communities)
    num_comms = max(communities) + 1
    N = g.vcount()

    if g.is_directed():
        adj = np.array(g.get_adjacency(attribute="weight" if g.is_weighted() else None).data, dtype=float)
        p = pagerank_uniform(adj, tau=tau)

        p_mod = np.zeros(num_comms)
        np.add.at(p_mod, communities, p)

        exit_flow = compute_exit_flow_nonuniform(g, communities, p)

        # recorded uniform teleportation: includes the teleportation term
        n_mod = np.bincount(communities, minlength=num_comms).astype(float)
        q_mod = tau * (1 - n_mod / N) * p_mod + (1 - tau) * exit_flow
    else:
        weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
        total_weight_x2 = 2 * np.sum(weights)
        strength = np.array(g.strength(weights="weight" if g.is_weighted() else None), dtype=float)
        p = strength / total_weight_x2

        p_mod = np.zeros(num_comms)
        np.add.at(p_mod, communities, p)

        exit_weights = compute_exit_weights(g, communities)
        q_mod = exit_weights / total_weight_x2

    q_sum = np.sum(q_mod)
    p_loop = p_mod + q_mod

    L = safe_xlogx(q_sum) - 2 * np.sum(safe_xlogx(q_mod)) \
        - np.sum(safe_xlogx(p)) + np.sum(safe_xlogx(p_loop))
    return L


# ===============================
# Smart unrecorded teleportation 
# ===============================

def pagerank_nonuniform(M, tau=0.15, tol=1e-15, maxiter=int(1e6)):
    """Two-step PageRank for smart unrecorded teleportation.
    Step 1: solve for p* with teleportation proportional to out-strength (Eq. 4).
    Step 2: one extra link-only step to get the recorded visit rates p (Eq. 6).
    """
    N = M.shape[0]
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    dangling = (row_sums == 0)
    no_incoming = (col_sums == 0)
    row_sums_safe = np.where(dangling, 1, row_sums)
    M_norm = M / row_sums_safe[:, None]

    # teleportation proportional to out-strength
    total_out = row_sums.sum()
    d = row_sums / total_out if total_out > 0 else np.ones(N) / N

    # Step 1: solve for p*
    p_star = np.ones(N) / N
    for _ in range(int(maxiter)):
        dangling_sum = p_star[dangling].sum()
        p_star_new = (1 - tau) * (p_star @ M_norm + dangling_sum * d) + tau * d
        if np.linalg.norm(p_star_new - p_star) < tol:
            p_star = p_star_new
            break
        p_star = p_star_new

    # Step 2: extra link-only step + dangling redistribution
    dangling_sum = p_star[dangling].sum()
    p = p_star @ M_norm + dangling_sum * d

    # Nodes with no incoming edges can only be reached via teleportation,
    # which isn't recorded, so their recorded flow is exactly 0
    p[no_incoming] = 0
    p = p / p.sum()
    return p


def compute_exit_flow_nonuniform(g, communities, p):
    """Rate of flow leaving each community via outgoing edges."""
    communities = np.array(communities)
    out_strength = np.array(g.strength(mode="out", weights="weight" if g.is_weighted() else None), dtype=float)
    weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
    edges = np.array(g.get_edgelist(), dtype=int)

    src, trg = edges[:, 0], edges[:, 1]
    src_com, trg_com = communities[src], communities[trg]
    betw = src_com != trg_com

    out_str_safe = np.where(out_strength > 0, out_strength, 1.0)
    flow = p[src] * weights / out_str_safe[src]

    exit_flow = np.zeros(max(communities) + 1)
    np.add.at(exit_flow, src_com[betw], flow[betw])
    return exit_flow


def compute_enter_flow_nonuniform(g, communities, p):
    """Rate of flow entering each community via incoming edges from outside."""
    communities = np.array(communities)
    out_strength = np.array(g.strength(mode="out", weights="weight" if g.is_weighted() else None), dtype=float)
    weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
    edges = np.array(g.get_edgelist(), dtype=int)

    src, trg = edges[:, 0], edges[:, 1]
    src_com, trg_com = communities[src], communities[trg]
    betw = src_com != trg_com

    out_str_safe = np.where(out_strength > 0, out_strength, 1.0)
    flow = p[src] * weights / out_str_safe[src]

    enter_flow = np.zeros(max(communities) + 1)
    np.add.at(enter_flow, trg_com[betw], flow[betw])
    return enter_flow


def compute_exit_weights(g, communities):
    """For undirected graphs: total weight of edges crossing community boundaries (both sides)."""
    communities = np.array(communities)
    weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
    edges = np.array(g.get_edgelist(), dtype=int)

    src_com = communities[edges[:, 0]]
    trg_com = communities[edges[:, 1]]
    betw = src_com != trg_com

    exit_weights = np.zeros(max(communities) + 1)
    np.add.at(exit_weights, src_com[betw], weights[betw])
    np.add.at(exit_weights, trg_com[betw], weights[betw])
    return exit_weights


def compute_description_length(g, communities, tau=0.15):
    """Map equation L using smart unrecorded teleportation for directed graphs,
    and the standard undirected formula for undirected graphs.
    For directed graphs, uses separate enter and exit flows per community
    (tutorial eqs. 9-10) to properly handle the asymmetry."""
    communities = np.array(communities)
    num_comms = max(communities) + 1
    N = g.vcount()

    if g.is_directed():
        adj = np.array(g.get_adjacency(attribute="weight" if g.is_weighted() else None).data, dtype=float)
        p = pagerank_nonuniform(adj, tau=tau)

        p_mod = np.zeros(num_comms)
        np.add.at(p_mod, communities, p)

        exit_flow = compute_exit_flow_nonuniform(g, communities, p)
        enter_flow = compute_enter_flow_nonuniform(g, communities, p)

        # enter flow for the index codebook, exit flow for the module codebook
        q_enter = enter_flow
        q_exit = exit_flow
    else:
        weights = np.array(g.es["weight"] if g.is_weighted() else [1.0] * g.ecount())
        total_weight_x2 = 2 * np.sum(weights)
        strength = np.array(g.strength(weights="weight" if g.is_weighted() else None), dtype=float)
        p = strength / total_weight_x2

        p_mod = np.zeros(num_comms)
        np.add.at(p_mod, communities, p)

        # for undirected, enter and exit are equal by symmetry
        exit_weights = compute_exit_weights(g, communities)
        q_enter = exit_weights / total_weight_x2
        q_exit = q_enter

    q_enter_sum = np.sum(q_enter)
    p_loop = p_mod + q_exit

    L = safe_xlogx(q_enter_sum) - np.sum(safe_xlogx(q_enter)) \
        - np.sum(safe_xlogx(q_exit)) \
        - np.sum(safe_xlogx(p)) + np.sum(safe_xlogx(p_loop))
    return L