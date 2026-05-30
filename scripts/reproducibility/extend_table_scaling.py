"""
extend_table_scaling.py — Empirical scaling beyond the N=3..5 paper benchmark.

Generates random connected weighted graphs at moderate density (E ~ 2.4*N) for
N in {6, 8, 10, 12}, K=3, integer weights in [1, 10]. For each (N, seed) pair
it builds all three oracle variants (Min Width, Min Depth, Balanced), transpiles
through Qiskit's preset pass manager (basis {cx,id,rz,sx,x}, optimization_level=3),
and prints qubits + transpiled depth. Two seeds per N to surface variance.

Run:  .venv/bin/python -u extend_table_scaling.py
"""
import math
import random
import sys
import time

import networkx as nx
from qiskit import transpile
from qiskit_aer import AerSimulator

from verify_table2 import (
    create_min_width_oracle, create_min_depth_oracle, create_balanced_oracle,
    classical_best_score, belletti_auto_chunk_size, BASIS_GATES,
)


def random_connected_weighted_graph(N: int, target_E: int, seed: int,
                                    w_lo: int = 1, w_hi: int = 10):
    """Return a list of (u, v, w) edges for a random connected graph."""
    rng = random.Random(seed)
    # Start with a random spanning tree (guarantees connectivity).
    nodes = list(range(N))
    rng.shuffle(nodes)
    edges_set = set()
    for i in range(1, N):
        u = nodes[i]
        v = nodes[rng.randrange(i)]
        edges_set.add((min(u, v), max(u, v)))
    # Add random extra edges until we hit target_E.
    while len(edges_set) < target_E:
        u, v = rng.sample(range(N), 2)
        edges_set.add((min(u, v), max(u, v)))
    return [(u, v, rng.randint(w_lo, w_hi)) for (u, v) in edges_set]


def main():
    print("\n=== Empirical scaling: random moderate-density graphs (K=3, weights in [1,10]) ===\n",
          flush=True)
    print(f"{'N':<3} {'E':<4} {'seed':<5} {'W_max':<6} {'S':<3} {'C*':<4} "
          f"{'MW_q':<5} {'MW_d':<7} {'MD_q':<5} {'MD_d':<7} {'Bal_q':<6} {'Bal_d':<7} {'t_sec':<6}",
          flush=True)
    print('-' * 95, flush=True)

    backend = AerSimulator()
    K = 3

    for N in [6, 8, 10, 12]:
        target_E = max(N - 1, math.ceil(2.4 * N))
        for seed in [11, 23]:
            edges = random_connected_weighted_graph(N, target_E, seed)
            E = len(edges)
            W_max = sum(w for _, _, w in edges)
            S = math.ceil(math.log2(W_max + 1)) + 1
            C = belletti_auto_chunk_size(E)
            t0 = time.time()
            target = classical_best_score(N, K, edges)

            qc_mw  = create_min_width_oracle(N, K, edges, target, 1)
            qc_md  = create_min_depth_oracle(N, K, edges, target, 1)
            qc_bal = create_balanced_oracle(N, K, edges, C, target, 1)

            t_mw  = transpile(qc_mw,  basis_gates=BASIS_GATES, optimization_level=3)
            t_md  = transpile(qc_md,  basis_gates=BASIS_GATES, optimization_level=3)
            t_bal = transpile(qc_bal, basis_gates=BASIS_GATES, optimization_level=3)
            elapsed = time.time() - t0

            print(f"{N:<3} {E:<4} {seed:<5} {W_max:<6} {S:<3} {C:<4} "
                  f"{qc_mw.num_qubits:<5} {t_mw.depth():<7} "
                  f"{qc_md.num_qubits:<5} {t_md.depth():<7} "
                  f"{qc_bal.num_qubits:<6} {t_bal.depth():<7} {elapsed:<6.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
