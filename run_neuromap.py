"""
run_neuromap.py

Runs Neuromap on the WikiCS graph (directed and undirected versions), with
4 underlying GNN architectures (GCN, GAT, GIN, GraphSAGE), 10 reruns each.

Hyperparameters are the results of a seperately carried out hyperparameter tuning

For every (graph variant x architecture x rerun), as soon as that run
finishes we save:
  - cluster_assignment.json: the hard community assignment
  - loss_curve.json: loss curve (bits)
  - a row appended to `overview.csv` in that variant's folder, with the
    architecture/settings, final codelength, runtime, and the filenames of
    the two files above

This is meant to be resumable and skip runs that were already carried out previously
(checked via trial_already_done() function)
Usage:
    python run_neuromap.py
"""

import sys
sys.path.append(".")
import time
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCN, GAT, GIN, GraphSAGE
import networkx as nx
import igraph as ig
from typing import Dict, Tuple, List, Union
from pathlib import Path


import src.neuromap as nm
from src.utils import (
    load_wikics_graph,
    atomic_write_json,
    append_csv_row,
    load_json,
    trial_already_done
)


####################################################################
# NEUROMAP CONFIG: placeholder hyperparameters (until we get params from marco)
####################################################################
ARCHITECTURES = {
    "GCN": GCN,
    "GAT": GAT,
    "GIN": GIN,
    "GraphSAGE": GraphSAGE,
}

NUM_RERUNS = 10
RESULTS_ROOT = "results/neuromap"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HYPERPARAMS = {
    "hidden_channels": 100,   # neuromap paper uses 4*sqrt(n)
    "num_layers": 2,
    "act": "selu",
    "norm": "batch",
    "dropout": 0.5,
    "lr": 1e-1,
    "epochs": 1000,
    "patience": 100,
}

########################################################################
# Helper functions
########################################################################

# helpers for building a PyG Data object straight from an igraph.Graph
# this is analogous to neuromap's own sparse_from_networkx
def sparse_from_igraph(G: ig.Graph) -> Tuple[torch.Tensor, Dict[int, int]]:
    has_names = "name" in G.vs.attributes()
    has_weight = "weight" in G.es.attributes()

    the_nodes = sorted(
        range(G.vcount()),
        key=lambda i: G.vs[i]["name"] if has_names else i
    )
    node_to_ID = {node: ID for (ID, node) in enumerate(the_nodes)}
    ID_to_node = {
        ID: (G.vs[node]["name"] if has_names else node)
        for (ID, node) in enumerate(the_nodes)
    }

    mode = "out" if G.is_directed() else "all"

    indices = [[], []]
    values = []
    for u in the_nodes:
        for v in sorted(G.neighbors(u, mode=mode)):
            weight = 1.0
            if has_weight:
                eid = G.get_eid(u, v, directed=G.is_directed())
                w = G.es[eid]["weight"]
                if w is not None:
                    weight = w
            indices[0].append(node_to_ID[u])
            indices[1].append(node_to_ID[v])
            values.append(float(weight))

    return (
        torch.sparse_coo_tensor(indices=indices, values=values,
                                 size=(len(the_nodes), len(the_nodes))),
        ID_to_node,
    )

# extension of neuromap's to_dataset() function, that also accepts igraph graphs
def to_dataset(G: Union[nx.Graph, ig.Graph], y_true: List[int]) -> Data:
    if isinstance(G, nx.Graph):
        adj = nm.sparse_from_networkx(G)[0]
    elif isinstance(G, ig.Graph):
        adj = sparse_from_igraph(G)[0]
    else:
        raise TypeError(f"G must be a networkx.Graph or igraph.Graph, got {type(G)}")

    data = Data()
    adj = adj.coalesce()
    data.edge_index = adj
    data.x = adj
    data.y = torch.Tensor(y_true).long()
    return data


# build model for given architecture & hyperparams
def build_model(architecture: str, n: int) -> torch.nn.Module:
    model_cls = ARCHITECTURES[architecture]
    return model_cls(
        in_channels=n,
        hidden_channels=HYPERPARAMS["hidden_channels"],
        num_layers=HYPERPARAMS["num_layers"],
        out_channels=int(np.sqrt(n)),  # as in the neuromap paper
        act=HYPERPARAMS["act"],
        norm=HYPERPARAMS["norm"],
        dropout=HYPERPARAMS["dropout"],
    )

# Neuromap clusterer with loss curve tracking
class Neuromap_with_loss_tracking(nm.Neuromap):
    """Same as nm.Neuromap, but records every epoch's train loss so we can
    save the full loss curve, not just the final/best value. Overrides the
    `fit` function of the original Neuromap/Clusterer class"""

    def fit(self, data: Data, epochs: int, patience: int, lr: float):
        self.data = data.to(device = self.device)
        self.pool = nm.MapEquationPooling(adj = data.edge_index, device = self.device)

        x = self.data.x.to_dense()

        l_best : float  = np.inf  # best loss
        s_best : torch.Tensor = None # best cluster

        optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        epoch          = 0
        no_improvement = 0

        loss_curve = [] # for tracking the losses over epochs

        while epoch < epochs and no_improvement < patience:
            self.train()
            optimizer.zero_grad()

            loss, s = self.forward(x = x)
            print(f"[Epoch {epoch:4}] L = {loss:.8f} bits")
            loss_curve.append(float(loss)) # add to loss curve list
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

        return l_best, s_best, loss_curve

# run a single neuromap clustering for a given graph, architecture
# we do multiple of these for each combo (hence the rerun_idx)
def run_single(g: ig.Graph, arch_name: str, rerun_idx: int, out_root: str) -> dict:
    run_dir = f"{out_root}/{arch_name}/run_{rerun_idx:02d}"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    assignment_path = f"{run_dir}/cluster_assignment.json" # for saving the cluset assignments
    loss_curve_path = f"{run_dir}/loss_curve.json" # for saving the loss curve

    # first check if this run already has been done 
    if trial_already_done(assignment_path):
        cached = load_json(assignment_path)
        return {
            "final_loss": cached["final_loss"],
            "runtime_seconds": cached["runtime_seconds"],
            "assignment_path": assignment_path,
            "loss_curve_path": loss_curve_path,
            "cached": True,
        }

    n = g.vcount()
    # this is a roundabout way of doing this (going from pytorch dataset
    # to igraph graph to pytorch dataset) but I do not want to put any brain 
    # power into isolating the GCC in the torch dataset
    data = to_dataset(G=g, y_true=[])

    model = build_model(arch_name, n)
    neuromap = Neuromap_with_loss_tracking(model=model, device=DEVICE)

    t0 = time.perf_counter()        
    final_loss, S, loss_curve = neuromap.fit(
        data,
        epochs=HYPERPARAMS["epochs"],
        patience=HYPERPARAMS["patience"],
        lr=HYPERPARAMS["lr"],
    )
    t_elapsed = time.perf_counter() - t0

    hard_clusters = nm.get_hard_clusters(S) if S is not None else []

    atomic_write_json(assignment_path, {
        "status": "completed",
        "architecture": arch_name,
        "rerun": rerun_idx,
        "final_loss": final_loss,
        "num_communities": int(len(set(hard_clusters))) if hard_clusters else 0,
        "runtime_seconds": t_elapsed,
        "communities": hard_clusters,
    })
    atomic_write_json(loss_curve_path, {
        "status": "completed",
        "architecture": arch_name,
        "rerun": rerun_idx,
        "loss_curve": loss_curve,
    })

    return {
        "final_loss": final_loss,
        "runtime_seconds": t_elapsed,
        "assignment_path": assignment_path,
        "loss_curve_path": loss_curve_path,
        "cached": False,
    }

# running the full sweep for either the directed or undirected WikiCS
def run_variant(directed: bool):
    variant = "directed" if directed else "undirected"
    print(f"\n=== WikiCS ({variant}) ===")

    g = load_wikics_graph(directed=directed)
    print(f"Loaded LCC: {g.vcount()} nodes, {g.ecount()} edges, directed={g.is_directed()}")

    variant_dir = f"{RESULTS_ROOT}/{variant}"
    Path(variant_dir).mkdir(parents=True, exist_ok=True)
    # overview for keeping track of the runs
    overview_path = f"{variant_dir}/overview.csv"
    overview_fields = [
        "architecture", "rerun", "hidden_channels", "num_layers", "out_channels",
        "act", "norm", "dropout", "lr", "epochs", "patience",
        "final_loss", "runtime_seconds", "cluster_assignment_file", "loss_curve_file", "cached",
    ]

    for arch_name in ARCHITECTURES:
        print(f"\n-- Architecture: {arch_name} --")
        for rerun in range(NUM_RERUNS):
            result = run_single(g, arch_name, rerun, variant_dir)
            append_csv_row(overview_path, overview_fields, {
                "architecture": arch_name,
                "rerun": rerun,
                "hidden_channels": HYPERPARAMS["hidden_channels"],
                "num_layers": HYPERPARAMS["num_layers"],
                "out_channels": int(np.sqrt(g.vcount())),
                "act": HYPERPARAMS["act"],
                "norm": HYPERPARAMS["norm"],
                "dropout": HYPERPARAMS["dropout"],
                "lr": HYPERPARAMS["lr"],
                "epochs": HYPERPARAMS["epochs"],
                "patience": HYPERPARAMS["patience"],
                "final_loss": result["final_loss"],
                "runtime_seconds": result["runtime_seconds"],
                "cluster_assignment_file": result["assignment_path"],
                "loss_curve_file": result["loss_curve_path"],
                "cached": result["cached"],
            })
            tag = " (cached)" if result["cached"] else ""
            print(f"  rerun {rerun:02d}: L = {result['final_loss']:.6f} bits, "
                  f"runtime = {result['runtime_seconds']:.1f}s{tag}")


def main():
    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    for directed in (True, False):
        run_variant(directed)


if __name__ == "__main__":
    main()
