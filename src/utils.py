import igraph as ig
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


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


def compare_partitions(comms1, comms2):
    """Small helper function to compare 2 community assignments for the same graph. 
    Prints number of communities, normalized mutual information, and adjusted Rand score.

    Args:
        comms1 (list[int] or numpy.ndarray): First community assignment to compare.
        comms2 (list[int] or numpy.ndarray): Second community assignment to compare.

    Raises:
        ValueError: _description_
    """
    if len(comms1) != len(comms2):
        raise ValueError(f"Partition shape mismatch: comms1 and comms2 have different lengths ({len(comms1)} and {len(comms2)}).")

    n_comms1 = len(set(comms1))
    n_comms2 = len(set(comms2))

    print(f"Comparing partitions:\nPartition 1: {n_comms1} communities\nPartition 2: {n_comms2} communities")

    nmi = normalized_mutual_info_score(comms1, comms2)
    ari = adjusted_rand_score(comms1, comms2)

    print(f"Normalized Mututal Information: {nmi:.4f}") # that one we know, between 0 and 1, if 1 -> identical partition
    # Rand score: label agreements/(label agreements + label disagreements), again, between 0 and 1, 1 -> identical partition
    # Adjusted rand score: "Adjusted for change": (RI - Expected_RI) / (max(RI) - Expected_RI)
    print(f"Adjusted Rand Index: {ari:.4f}")  # between -0.5 and 1.0, 0 -> random, 1.0 -> identical

    #return nmi, jaccard, ari