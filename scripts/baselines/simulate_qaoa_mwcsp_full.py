"""
MWCSP QAOA Benchmark (Full Formulation)
========================================
Formulates the Maximum Weight Colorable Subgraph Problem (MWCSP)
as a proper QUBO/Ising Hamiltonian with:
  - One-Hot Encoding: K qubits per node  =>  K*N qubits total
  - Penalty Term:  lambda * sum_i ( 1 - sum_k x_{i,k} )^2   (one color per node)
  - Cost Term:     sum_{(i,j) in E} w_ij * sum_k x_{i,k} * x_{j,k}  (penalize same-color edges)

QAOA minimizes H_C = H_penalty + H_cost
Success Rate = fraction of shots that decode to the EXACT globally optimal valid coloring
"""

import numpy as np
import random
import itertools
import time
from collections import defaultdict

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_random_graph(n_nodes, edge_prob=0.6, w_min=1, w_max=10, seed=None):
    """Return (nodes, edges) where edges = list of (i, j, weight)."""
    rng = random.Random(seed)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                w = rng.randint(w_min, w_max)
                edges.append((i, j, w))
    return list(range(n_nodes)), edges


def brute_force_mwcsp(n, K, edges):
    """Enumerate all K^N colorings; return (best_score, best_coloring)."""
    best_score = -1
    best_col   = None
    for coloring in itertools.product(range(K), repeat=n):
        score = sum(w for (i, j, w) in edges if coloring[i] != coloring[j])
        if score > best_score:
            best_score = score
            best_col   = coloring
    return best_score, best_col


# ─────────────────────────────────────────────
#  QUBO / Ising Construction
# ─────────────────────────────────────────────

def qubit_idx(node, color, K):
    """Map (node, color) -> qubit index in one-hot encoding."""
    return node * K + color


def build_ising(n, K, edges, lam):
    """
    Build Ising coefficients (h, J) from MWCSP QUBO.

    QUBO:
      H = lam * sum_i ( 1 - sum_k x_{i,k} )^2
          + sum_{(i,j,w)} w * sum_k x_{i,k} * x_{j,k}

    x_{i,k} in {0,1} mapped via  x = (1 - Z) / 2

    Returns:
      h : dict  qubit -> coefficient   (linear)
      J : dict  (q1,q2) -> coefficient (quadratic, q1<q2)
    """
    num_q = n * K
    Q = np.zeros((num_q, num_q))

    # --- Penalty term: lam*(1 - sum_k x_{i,k})^2 for each node i ---
    for i in range(n):
        for ka in range(K):
            qa = qubit_idx(i, ka, K)
            Q[qa, qa] += -lam            # diagonal: -lam * x_{i,ka}
            for kb in range(ka + 1, K):
                qb = qubit_idx(i, kb, K)
                Q[qa, qb] += 2 * lam     # cross: +2*lam * x_{i,ka}*x_{i,kb}

    # --- Cost term: +w * x_{i,k}*x_{j,k}  (same color on connected nodes is bad) ---
    for (i, j, w) in edges:
        for k in range(K):
            qi = qubit_idx(i, k, K)
            qj = qubit_idx(j, k, K)
            q_lo, q_hi = min(qi, qj), max(qi, qj)
            Q[q_lo, q_hi] += w

    # Convert QUBO -> Ising via  x_q = (1 - z_q)/2
    # H_ising = const + sum_q h_q z_q + sum_{q<r} J_{qr} z_q z_r
    h = {}
    J = {}
    for q in range(num_q):
        val = -Q[q, q] / 2.0
        if abs(val) > 1e-9:
            h[q] = h.get(q, 0) + val
        for r in range(q + 1, num_q):
            val_J = Q[q, r] / 4.0
            if abs(val_J) > 1e-9:
                J[(q, r)] = J.get((q, r), 0) + val_J
            val_h_q = -Q[q, r] / 4.0
            val_h_r = -Q[q, r] / 4.0
            if abs(val_h_q) > 1e-9:
                h[q] = h.get(q, 0) + val_h_q
            if abs(val_h_r) > 1e-9:
                h[r] = h.get(r, 0) + val_h_r

    return h, J


# ─────────────────────────────────────────────
#  QAOA Circuit
# ─────────────────────────────────────────────

def build_qaoa_circuit(num_q, h, J, p):
    """Build QAOA ansatz with p layers."""
    gamma = ParameterVector('g', p)
    beta  = ParameterVector('b', p)

    qc = QuantumCircuit(num_q)
    qc.h(range(num_q))   # uniform superposition

    for layer in range(p):
        # --- Cost unitary  U_C(gamma) ---
        for q, coeff in h.items():
            qc.rz(2 * gamma[layer] * coeff, q)
        for (q1, q2), coeff in J.items():
            qc.cx(q1, q2)
            qc.rz(2 * gamma[layer] * coeff, q2)
            qc.cx(q1, q2)
        # --- Mixer unitary  U_B(beta) ---
        for q in range(num_q):
            qc.rx(2 * beta[layer], q)

    qc.measure_all()
    return qc, gamma, beta


def run_qaoa(num_q, h, J, p, shots=2048, max_iter=200):
    """Optimize QAOA parameters and return measurement counts."""
    simulator = AerSimulator(method='statevector')
    qc, gamma, beta = build_qaoa_circuit(num_q, h, J, p)

    def cost_fn(params):
        g_vals = params[:p]
        b_vals = params[p:]
        bound = qc.assign_parameters(
            {gamma[i]: g_vals[i] for i in range(p)} |
            {beta[i]:  b_vals[i] for i in range(p)}
        )
        job    = simulator.run(bound, shots=1024)
        counts = job.result().get_counts()
        # Estimate expectation of Ising cost (lower = better for COBYLA)
        total = 0
        total_shots = sum(counts.values())
        for bitstring, cnt in counts.items():
            z = np.array([1 - 2 * int(b) for b in reversed(bitstring)])
            e = sum(coeff * z[q] for q, coeff in h.items())
            e += sum(coeff * z[q1] * z[q2] for (q1, q2), coeff in J.items())
            total += e * cnt
        return total / total_shots

    rng = np.random.default_rng(42)
    x0  = rng.uniform(-np.pi, np.pi, 2 * p)
    res = minimize(cost_fn, x0, method='COBYLA',
                   options={'maxiter': max_iter, 'rhobeg': 0.5})

    # Final measurement with best params
    best_g = res.x[:p]
    best_b = res.x[p:]
    bound  = qc.assign_parameters(
        {gamma[i]: best_g[i] for i in range(p)} |
        {beta[i]:  best_b[i] for i in range(p)}
    )
    job    = simulator.run(bound, shots=shots)
    counts = job.result().get_counts()
    return counts


# ─────────────────────────────────────────────
#  Decode & Score
# ─────────────────────────────────────────────

def decode_one_hot(bitstring, n, K):
    """
    Decode one-hot bitstring -> coloring list (or None if invalid).
    bitstring[0] = qubit 0 (little-endian from Qiskit).
    """
    bits = [int(b) for b in reversed(bitstring)]  # q0 first
    coloring = []
    for i in range(n):
        chunk = bits[i * K:(i + 1) * K]
        ones  = [k for k, v in enumerate(chunk) if v == 1]
        if len(ones) != 1:
            return None           # invalid: 0 or >1 color
        coloring.append(ones[0])
    return tuple(coloring)


def coloring_weight(coloring, edges):
    """Sum weights of properly colored (different-color) edges."""
    if coloring is None:
        return 0
    return sum(w for (i, j, w) in edges if coloring[i] != coloring[j])


# ─────────────────────────────────────────────
#  Main Benchmark
# ─────────────────────────────────────────────

def run_benchmark(K_list=(3, 5), N_list=(3, 4, 5), num_graphs=10, p=3, shots=2048):
    header = f"{'K':>3} | {'N':>3} | {'Graph':>6} | {'Lambda':>8} | " \
             f"{'Best W':>7} | {'Alpha':>7} | {'SuccRate%':>10} | {'Valid%':>7} | {'Time(s)':>8}"
    print(header)
    print("-" * len(header))

    results = defaultdict(list)

    for K in K_list:
        # penalty should dominate: set lam > max possible weight per node
        lam = 15.0  # works well for w in [1,10]

        for N in N_list:
            for g_idx in range(num_graphs):
                seed = K * 1000 + N * 100 + g_idx
                _, edges = make_random_graph(N, edge_prob=0.7, seed=seed)

                if not edges:
                    continue

                # Classical optimal
                t0 = time.time()
                opt_score, opt_col = brute_force_mwcsp(N, K, edges)

                # Build Ising
                num_q = N * K
                h_ising, J_ising = build_ising(N, K, edges, lam)

                # Run QAOA
                try:
                    counts = run_qaoa(num_q, h_ising, J_ising, p=p, shots=shots)
                except Exception as ex:
                    print(f"  ERROR K={K} N={N} g={g_idx}: {ex}")
                    continue

                elapsed = time.time() - t0

                # Analyse counts
                total_shots = sum(counts.values())
                success_shots = 0
                valid_shots   = 0
                weighted_sum  = 0.0

                for bitstring, cnt in counts.items():
                    col = decode_one_hot(bitstring, N, K)
                    if col is None:
                        continue        # invalid one-hot
                    valid_shots += cnt
                    w = coloring_weight(col, edges)
                    weighted_sum += w * cnt
                    if col == opt_col:
                        success_shots += cnt

                valid_pct   = 100.0 * valid_shots / total_shots if total_shots else 0
                success_pct = 100.0 * success_shots / total_shots if total_shots else 0
                alpha       = (weighted_sum / valid_shots / opt_score) if (valid_shots and opt_score) else 0.0

                results[(K, N)].append({
                    'alpha': alpha,
                    'success_pct': success_pct,
                    'valid_pct': valid_pct,
                })

                print(f"{K:>3} | {N:>3} | {g_idx:>6} | {lam:>8.1f} | "
                      f"{opt_score:>7} | {alpha:>7.3f} | {success_pct:>10.2f} | "
                      f"{valid_pct:>7.1f} | {elapsed:>8.2f}")

    # ── Summary Table ──
    print("\n\n" + "="*75)
    print("SUMMARY  (averaged over graphs)")
    print("="*75)
    print(f"{'K':>3} | {'N':>3} | {'Avg Alpha':>10} | {'Avg Success%':>13} | {'Avg Valid%':>11}")
    print("-"*55)
    for (K, N), recs in sorted(results.keys()):
        recs = results[(K, N)]
        avg_a = np.mean([r['alpha']       for r in recs])
        avg_s = np.mean([r['success_pct'] for r in recs])
        avg_v = np.mean([r['valid_pct']   for r in recs])
        print(f"{K:>3} | {N:>3} | {avg_a:>10.3f} | {avg_s:>13.2f} | {avg_v:>11.1f}")
    print("="*75)


if __name__ == "__main__":
    print("=" * 75)
    print("  MWCSP QAOA Benchmark  —  Full One-Hot Formulation (K ∈ {3, 5})")
    print("  QAOA layers p=3 | 10 random graphs per (K,N) | shots=2048")
    print("=" * 75 + "\n")

    run_benchmark(
        K_list=(3, 5),
        N_list=(3, 4, 5),
        num_graphs=10,
        p=3,
        shots=2048
    )
