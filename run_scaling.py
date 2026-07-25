"""
run_infomap.py

Runs 3 infomap community detection methods on the WikiCS graph,
for both its directed and undirected version:
  1. "custom"      - our own implementation (src/optimize.py)
  2. "igraph"       - igraph's community_infomap()
  3. "infomap_pkg"  - the `infomap` package's impementation

Usage:
    python run_infomap.py
"""

import sys
import time
import numpy as np
import igraph as ig
import infomap
import networkx as nx
from typing import Dict, Tuple, List, Union


import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCN, GAT, GIN, GraphSAGE

from pathlib import Path
sys.path.append(".")  # so `import src....` works when run from project root

import src.optimize as opt
import src.map_equation as meq
import src.utils as ut
import src.neuromap as nm

# network params
NODE_NUMBERS = np.logspace(1.5, 4, 11, dtype=int)
COMMS = 5
P_IN = 0.25
P_OUT = 0.05

NUM_TRIALS = 3
INFOMAP_RESULTS_ROOT = "results/scaling/infomap"
NEUROMAP_RESULTS_ROOT = "results/scaling/neuromap"

# specifics for the custom infomap:
CUSTOM_MAX_ITER = 1000
CUSTOM_TELEPORTATION = "uniform"


# specifics for neuromap
ARCHITECTURES = {
    "GCN": GCN,
    "GAT": GAT,
    "GIN": GIN,
    "GraphSAGE": GraphSAGE,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HYPERPARAMS = {
    "num_layers": 2,
    "act": "selu",
    "norm": "batch",
    "dropout": 0.5,
    "lr": 1e-3,
    "epochs": 1000,
    "patience": 100,
}


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
        hidden_channels=int(4*np.sqrt(n)), # as in paper
        num_layers=HYPERPARAMS["num_layers"],
        out_channels=int(np.sqrt(n)),
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
                    l_best = float(loss)
                    no_improvement = 0
                    s_best = s
                else:
                    no_improvement += 1

                epoch += 1

        return l_best, s_best, loss_curve

# run a single neuromap clustering for a given graph, architecture
# we do multiple of these for each combo 
def run_single(g: ig.Graph, arch_name: str, out_root: str) -> dict:
    run_dir = f"{out_root}/{arch_name}"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    assignment_path = f"{run_dir}/cluster_assignment.json" # for saving the cluset assignments
    loss_curve_path = f"{run_dir}/loss_curve.json" # for saving the loss curve

    # first check if this run already has been done 
    if ut.trial_already_done(assignment_path):
        cached = ut.load_json(assignment_path)
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

    ut.atomic_write_json(assignment_path, {
        "status": "completed",
        "architecture": arch_name,
        "final_loss": final_loss,
        "num_communities": int(len(set(hard_clusters))) if hard_clusters else 0,
        "runtime_seconds": t_elapsed,
        "communities": hard_clusters,
    })

    ut.atomic_write_json(loss_curve_path, {
        "status": "completed",
        "architecture": arch_name,
        "loss_curve": loss_curve,
    })

    return {
        "final_loss": final_loss,
        "runtime_seconds": t_elapsed,
        "assignment_path": assignment_path,
        "loss_curve_path": loss_curve_path,
        "cached": False,
    }


# running the different infomap methods:
def run_custom(g: ig.Graph, out_dir: str):
    """
    Run custom infomap implementation on the input graph, write result to json
    & return results.
    """
    path = f"{out_dir}/custom_results.json"
    t0 = time.perf_counter()
    communities = opt.search_community_partition(
        g,
        num_restarts=NUM_TRIALS,
        max_iter=CUSTOM_MAX_ITER,
        teleportation=CUSTOM_TELEPORTATION,
        verbose=False,
        )
    t_elapsed = time.perf_counter() - t0
    codelength = meq.compute_description_length(g, communities, teleportation=CUSTOM_TELEPORTATION)

    result_dict = {
        "method": "custom", 
        "num_trials": NUM_TRIALS,
        "codelength": float(codelength),
        "num_communities": int(len(np.unique(communities))),
        "runtime_seconds": t_elapsed,
        "communities": list(map(int, communities)),
    }

    ut.atomic_write_json(path, result_dict)
    return result_dict


def run_igraph(g: ig.Graph, out_dir: str):
    """
    Run igraph package's infomap on the input graph, write result to json
    & return results.
    """
    path = f"{out_dir}/igraph_results.json"

    t0 = time.perf_counter()
    clustering = g.community_infomap(trials=NUM_TRIALS)
    t_elapsed = time.perf_counter() - t0

    communities = clustering.membership
    codelength = clustering.codelength

    result_dict = {
        "method": "igraph",
        "num_trials": NUM_TRIALS,
        "codelength": float(codelength),
        "num_communities": int(len(np.unique(communities))),
        "runtime_seconds": t_elapsed,
        "communities": list(map(int, communities)),
    }

    # write to a file immediately just to be sure
    ut.atomic_write_json(path, result_dict)
    return result_dict


def run_infomap_pkg(g: ig.Graph, directed: bool, out_dir: str):
    """
    Run infomap package's infomap on the input graph, write result to json
    & return results.
    """
    path = f"{out_dir}/infomap_pkg_results.json"  # single file: this method's one run covers all trials

    t0 = time.perf_counter()
    result = infomap.run(
        g,
        two_level=True,
        directed=directed,
        num_trials=NUM_TRIALS
    )
    t_elapsed = time.perf_counter() - t0

    codelength = float(result.codelength)
    communities = [mod for _, mod in sorted(result.modules().items())]

    result_dict = {
        "method": "infomap_pkg",
        "num_trials": NUM_TRIALS,
        "codelength": codelength,
        "num_communities": int(len(np.unique(communities))),
        "runtime_seconds": t_elapsed,
        "communities": list(map(int, communities)),
    }
    # write to a file immediately just to be sure
    ut.atomic_write_json(path, result_dict)
    return result_dict


# run all methods for directed/undirected variant of WikiCS:
def run_variant(directed: bool):
    t_start = time.perf_counter()
    variant = "directed" if directed else "undirected"
    for N in NODE_NUMBERS:
        print(f"\n=== Size {N} network ({variant}) ===")
        g = ut.generate_sbm(N, COMMS, P_IN, P_OUT, directed=directed, weighted=False)
        graph_dict = {"comms_true":  list(map(int, g.vs["community"])),
                      "p_in": P_IN,
                      "p_out": P_OUT,
                      "num_comms": COMMS}

        print(f"Generated graph: {g.vcount()} nodes, {g.ecount()} edges, directed={g.is_directed()}")

        # INFOMAP
        summary_dir = f"{INFOMAP_RESULTS_ROOT}/{variant}"
        Path(summary_dir).mkdir(parents=True, exist_ok=True)
        ut.atomic_write_json(summary_dir + f"/graph{N}.json", graph_dict)

        dir = summary_dir + f"/N{N}"
        Path(dir).mkdir(parents=True, exist_ok=True)

        # run igraph
        print("Running igraph infomap ...")
        res_ig = run_igraph(g, dir)
        res_ig["N"]= N
        ut.append_csv_row(f"{dir}/summary.csv", res_ig.keys(), res_ig)
        print(f"[igraph]: L = {res_ig['codelength']:.6f} bits")

        # run infomap package
        print("Running infomap pkg ...")
        res_im = run_infomap_pkg(g, directed, dir)
        res_im["N"]= N
        ut.append_csv_row(f"{dir}/summary.csv", res_im.keys(), res_im)
        print(f"[infomap pkg]: L = {res_im['codelength']:.6f} bits")

        # run custom
        print("Running custom infomap ...")
        res_custom = run_custom(g, dir)
        res_custom["N"]=N
        ut.append_csv_row(f"{dir}/summary.csv", res_custom.keys(), res_custom)
        print(f"[custom]: L = {res_custom['codelength']:.6f} bits")
        
        runtime = time.perf_counter() - t_start
        print(f"Runtime for N={N}, {variant}: {runtime:.2f}")


        # NEUROMAP
        variant_dir = f"{NEUROMAP_RESULTS_ROOT}/{variant}"
        Path(variant_dir).mkdir(parents=True, exist_ok=True)
        # overview for keeping track of the runs
        overview_path = f"{variant_dir}/overview.csv"
        overview_fields = [ "N",
            "architecture", "hidden_channels", "num_layers", "out_channels",
            "act", "norm", "dropout", "lr", "epochs", "patience",
            "final_loss", "runtime_seconds", "cluster_assignment_file", "loss_curve_file", "cached",
        ]
    
        for arch_name in ARCHITECTURES:
            print(f"\n-- Architecture: {arch_name} --")
            result = run_single(g, arch_name, variant_dir+f"/N{N}")
            ut.append_csv_row(overview_path, overview_fields, {
                "N": N,
                "architecture": arch_name,
                "hidden_channels": int(4*np.sqrt(N)),
                "num_layers": HYPERPARAMS["num_layers"],
                "out_channels": int(np.sqrt(N)),
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
            print(f" L = {result['final_loss']:.6f} bits, "
                    f"runtime = {result['runtime_seconds']:.1f}s{tag}")


def main():
    Path(INFOMAP_RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(NEUROMAP_RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    for directed in (True, False):
        run_variant(directed)

if __name__ == "__main__":
    main()
