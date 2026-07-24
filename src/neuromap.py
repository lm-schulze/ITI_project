'''
This contains all the relevant helpers from the reference Neuromap implementation
Everthing is taken from the full implementation in this repository:
https://github.com/chrisbloecker/neuromap

With the corresponding publication being
(TODO: PROPERLY FORMAT THIS)

@misc{blöcker2024mapequationgoesneural,
  title         = {The Map Equation Goes Neural: Mapping Network Flows with Graph Neural Networks},
  author        = {Christopher Bl\"cker and Chester Tan and Ingo Scholtes},
  year          = {2024},
  publisher     = {arXiv},
  doi           = {10.48550/arXiv.2310.01144},
  url           = {https://doi.org/10.48550/arXiv.2310.01144},
  eprint        = {2310.01144},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  howpublished  = {\href{https://doi.org/10.48550/arXiv.2310.01144}{arXiv:2310.01144}}
}
'''

from abc                  import abstractmethod
from numpy                import inf
from torch                import Tensor
from torch.nn             import Parameter
from torch.nn.functional  import softmax, sigmoid
from torch_geometric.nn   import GCN
from torch_geometric.data import Data
from typing               import Dict, List, Tuple

import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn           as sb
import scipy


import torch_geometric.utils as utils
import torch_geometric.transforms as T
import igraph as ig
import networkx as nx
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx


# plotting/visualisation helpers:
def plot_overlapping(G, S, eps = 1e-3, pos = None, figsize = (5,5), node_scale = 1.0, node_palette = sb.color_palette("colorblind"), link_palette = sb.color_palette("pastel")):
    if pos is None:
        pos = nx.kamada_kawai_layout(G)
        pos = { k:(x,-y) for k,(x,y) in pos.items() }

    A = torch.tensor(nx.adjacency_matrix(G, nodelist = sorted(G.nodes)).todense(), dtype=torch.float)
    p = torch.sum(A, dim = 1) / torch.sum(A)

    existing_modules = []

    for module_ix, total_assignments in enumerate(torch.sum(S,0).numpy()):
        if total_assignments > eps:
            existing_modules.append(module_ix)

    S_reduced = S[:,existing_modules]

    fig, ax = plt.subplots(1, 1, figsize = figsize)

    modules = dict()

    for ix,node in enumerate(sorted(G.nodes())):
        assignment = np.array([s if s > eps else 0 for s in S_reduced[ix].numpy()])
        modules[node] = assignment / sum(assignment)

    edgelist     = []
    edge_colours = []
    edge_widths  = []
    for (u,v) in G.edges:
        edgelist.append((u,v))
        edge_colours.append([link_palette[m % len(link_palette)] for m,s in enumerate(modules[u]) if s > 0][0] if all(modules[u] == modules[v]) else "grey")
        d = G.get_edge_data(u,v)
        edge_widths.append(d["weight"] if "weight" in d else 1)
        
    nx.draw_networkx_edges( G = G
                          , pos = pos
                          , nodelist = sorted(G.nodes)
                          , edgelist = edgelist
                          , width = edge_widths
                          , edge_color = edge_colours
                          , ax = ax
                          , min_source_margin = 1
                          , min_target_margin = 1
                          , arrows = True
                          , connectionstyle = "arc3,rad=0.1"
                          )

    for ix,node in enumerate(sorted(G.nodes())):
        assignment = modules[node]
        ax.pie( [s for s in assignment if s > 0] # flows
              , colors = [node_palette[m % len(node_palette)] for m,s in enumerate(assignment) if s > 0]
              , center = pos[node]
              , radius = 0.25 * np.sqrt(float(p[ix])) * node_scale
              , startangle = 0 # startangle
              , wedgeprops = { "linewidth": 1, "edgecolor": "white" }
              )
    plt.autoscale()

    plt.tight_layout()


def plot_S(S, G):
    fig, ax = plt.subplots(1,1,figsize=(6,5))
    sb.heatmap(S, cmap = sb.color_palette("viridis", as_cmap=True), ax = ax)
    ax.set_xlabel("module")
    ax.set_ylabel("node")
    plt.show()


def get_hard_clusters(S):
    hard = S.argmax(dim = 1).cpu().numpy()
    y    = []
    cluster_to_ID = dict((cluster,ix) for ix,cluster in enumerate(set(hard)))
    for cluster in hard:
        y.append(cluster_to_ID[cluster])
    return y


def to_dataset(G: nx.Graph, y_true: List[int]) -> Data:
    """
    Takes a networkx graph and a list of community labels for the nodes and
    returns them as a pyg Data representation.

    Parameters
    ----------
    G : nx.Graph
        The networkx graph.

    y_true : List[int]
        List of the nodes' community labels.

    Returns
    -------
    Data
        A Data object where the edge index and node features X are a sparse
        tensor representation of the graph's adjacency matrix and the node
        labels a the nodes' true communities.
    """
    data = Data()
    data.edge_index = sparse_from_networkx(G)[0].coalesce()
    data.x          = sparse_from_networkx(G)[0].coalesce()
    data.y          = torch.Tensor(y_true).long()

    return data


# create smart teleportation flow matrix and flow distribution as described
#  - https://arxiv.org/abs/2311.04036
#  - https://www.nature.com/articles/ncomms5630
def mkSmartTeleportationFlow(A, alpha = 0.15, iter = 1000, device : str = "cpu"):
    # build the transition matrix
    T = torch.nan_to_num(A.T * (torch.sum(A, 1)**(-1.0)).to_dense(), nan = 0.0).T.to(device = device)

    # distribution according to nodes' in-degrees
    e_v = (torch.sum(A, dim = 0) / torch.sum(A)).to_dense().to(device = device)

    # calculate the flow distribution with a power iteration
    p = e_v
    for _ in range(iter):
        p = alpha * e_v + (1-alpha) * p @ T
    
    # make the flow matrix for minimising the map equation
    F = alpha * A / torch.sum(A) + (1-alpha) * (p * T.T).T
    
    return F, p


# We encode the map equation loss in a pooling operator, but are actually
# only interested in the codelength for now
class MapEquationPooling(torch.nn.Module):
    def __init__(self, adj: Tensor, device : str = "cpu", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.adj       = adj
        self.F, self.p = mkSmartTeleportationFlow(self.adj, device = device)

        # this term is constant, so only calculate it once
        self.p_log_p = torch.sum(self.p * torch.nan_to_num(torch.log2(self.p), nan = 0.0))

    def forward(self, x, s):
        C      = s.T @ self.F @ s
        diag_C = torch.diag(C)

        q   = 1.0 - torch.trace(C)
        q_m = torch.sum(C, dim = 1) - diag_C
        m_exit = torch.sum(C, dim = 0) - diag_C
        p_m = q_m + torch.sum(C, dim = 0)

        codelength = torch.sum(q      * torch.nan_to_num(torch.log2(q),      nan = 0.0)) \
                   - torch.sum(q_m    * torch.nan_to_num(torch.log2(q_m),    nan = 0.0)) \
                   - torch.sum(m_exit * torch.nan_to_num(torch.log2(m_exit), nan = 0.0)) \
                   - self.p_log_p \
                   + torch.sum(p_m    * torch.nan_to_num(torch.log2(p_m),    nan = 0.0))

        x_pooled   = torch.matmul(s.T, x)
        adj_pooled = s.T @ self.adj @ s

        return x_pooled, adj_pooled, codelength
    

def sparse_from_networkx(G : nx.Graph) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Converts a networkx graph to a sparse tensor.

    Parameters
    ----------
    G : nx.Graph
        The networkx graph, which can be weighted and/or directed.

    Returns
    ------
    Tuple[torch.Tensor, Dict[int, int]]
        A tuple containing the sparse tensor representation of the input graph
        and a dictionary from zero-based IDs to the original node names.
    """

    # always make sure to sort the nodes so they're in the expected order
    the_nodes  = list(sorted(G.nodes))
    node_to_ID = { node:ID for (ID,node) in enumerate(the_nodes) }
    ID_to_node = { ID:node for (ID,node) in enumerate(the_nodes) }

    indices = [[],[]]
    values  = []
    for u in the_nodes:
        for v in sorted(G.neighbors(u)): # again, always sorting...
            weight = 1.0
            data   = G.get_edge_data(u, v)
            if "weight" in data:
                weight = data["weight"]
            indices[0].append(node_to_ID[u])
            indices[1].append(node_to_ID[v])
            values.append(float(weight))
    
    return ( torch.sparse_coo_tensor( indices = indices
                                    , values  = values
                                    , size    = (len(the_nodes), len(the_nodes))
                                    )
           , ID_to_node
           )


# A clusterer that runs the optimisation.
# The missing ingredient: a specific forward method (next cell).
class Clusterer(torch.nn.Module):
    def __init__(self, model, device: str = "cpu") -> None:
        super().__init__()

        self.model  = model
        self.device = device

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError(f"forward not implemented on {self._get_name()}")

    def fit(self, data: Data, epochs: int, patience: int, lr: float):
        self.data = data.to(self.device)
        x = self.data.x.to_dense()

        l_best : float  = inf  # best loss
        s_best : Tensor = None # best cluster

        optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        epoch          = 0
        no_improvement = 0

        while epoch < epochs and no_improvement < patience:
            self.train()
            optimizer.zero_grad()

            loss, s = self.forward(x = x)
            print(f"[Epoch {epoch:4}] L = {loss:.8f} bits")
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                loss, s = self.forward(x = x)

                if loss < l_best:
                    l_best         = float(loss)
                    no_improvement = 0
                    s_best         = s
                else:
                    no_improvement += 1

                epoch += 1


        return l_best, s_best
    

# Neuromap inherits from Clusterer and defines the specific forward method.
class Neuromap(Clusterer):
    def __init__(self, model, device: str = "cpu") -> None:
        super().__init__(model = model, device = device)

        # softmax temperature
        self.t = Parameter(torch.zeros(1)).to(device = self.device)


    def forward(self, x):
        s = softmax(self.model(x, self.data.edge_index) / sigmoid(self.t), dim = 1)

        if self.training:
            s = s + 1e-8

        _, _, loss = self.pool(x = x, s = s)

        return loss, s


    def fit(self, data: Data, epochs: int, patience: int, lr: float):
        self.data = data.to(device = self.device)
        self.pool = MapEquationPooling(adj = data.edge_index, device = self.device)
        
        return super().fit(data = data, epochs = epochs, patience = patience, lr = lr)