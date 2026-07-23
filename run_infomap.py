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

from pathlib import Path


sys.path.append(".")  # so `import src....` works when run from project root

import src.optimize as opt
import src.map_equation as meq
import src.utils as ut


NUM_TRIALS = 10
RESULTS_ROOT = "results/infomap"
# specifics for the custom infomap:
CUSTOM_MAX_ITER = 1000
CUSTOM_TELEPORTATION = "uniform"



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
        num_trials=NUM_TRIALS,
        seed=0,
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
    print(f"\n=== WikiCS ({variant}) ===")

    g = ut.load_wikics_graph(directed=directed)
    print(f"Loaded LCC: {g.vcount()} nodes, {g.ecount()} edges, directed={g.is_directed()}")

    dir = f"{RESULTS_ROOT}/{variant}"
    Path(dir).mkdir(parents=True, exist_ok=True)

    # run custom
    print("Running custom infomap ...")
    res_custom = run_custom(g, dir)
    ut.append_csv_row(f"{dir}/summary.csv", res_custom.keys(), res_custom)
    print(f"[custom]: L = {res_custom['codelength']:.6f} bits")

    # run igraph
    print("Running igraph infomap ...")
    res_ig = run_igraph(g, dir)
    ut.append_csv_row(f"{dir}/summary.csv", res_ig.keys(), res_ig)
    print(f"[igraph]: L = {res_ig['codelength']:.6f} bits")

    # run infomap package
    print("Running infomap pkg ...")
    res_im = run_infomap_pkg(g, dir)
    ut.append_csv_row(f"{dir}/summary.csv", res_im.keys(), res_im)
    print(f"[custom]: L = {res_im['codelength']:.6f} bits")

    runtime = time.perf_counter() - t_start
    print(f"Runtime for {variant} WikiCS (s): {runtime:.2f}")


def main():
    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    for directed in (True, False):
        run_variant(directed)

if __name__ == "__main__":
    main()