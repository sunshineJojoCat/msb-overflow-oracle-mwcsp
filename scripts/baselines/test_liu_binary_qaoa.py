"""
Liu et al. Binary-Encoded QAOA — Adapted for Weighted MWCSP Benchmark
======================================================================
Based on: "Efficient hybrid variational quantum algorithm for solving
graph coloring problem" (arXiv: 2504.21335, April 2025)

Adapts Liu's bit-wise Binary QAOA to weighted edges for direct
comparison with our MSB-Overflow Grover architecture.

Simulation: Exact Statevector (noise-free, best-case for QAOA)
Optimizer:  L-BFGS-B with multi-seed restarts
"""
import math, time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from scipy.optimize import minimize

# ── Config ──────────────────────────────────────────────────────
K = 3
M = math.ceil(math.log2(K))          # 2 bits per node
INVALID_PENALTY = 10.0               # penalty for binary state >= K
P_LAYERS = 7
NUM_SEEDS = 10

# ── Topologies (same as paper) ──────────────────────────────────
TOPOS = {
    "Weighted Triangle": {"N": 3, "edges": [(0,1,1),(1,2,2),(2,0,3)]},
    "Heavy Star S4":     {"N": 4, "edges": [(0,1,4),(0,2,2),(0,3,5)]},
    "Diagonal Square":   {"N": 4, "edges": [(0,1,2),(1,2,3),(2,3,2),(3,0,3),(0,2,10)]},
    "Frustrated K4":     {"N": 4, "edges": [(0,1,4),(0,2,5),(0,3,3),(1,2,6),(1,3,2),(2,3,4)]},
    "Path P5":           {"N": 5, "edges": [(0,1,3),(1,2,7),(2,3,2),(3,4,6)]},
}

# ── Ising Builder (Liu bit-wise + weighted) ─────────────────────
def build_liu_ising(N, edges):
    """
    Collision: w * sum_b (1 + Z_{ub} Z_{vb}) / 2   (penalise matching bits)
    Invalid:   A * x_{i0} x_{i1}  for each node     (penalise state '11')
    """
    h, J = {}, {}

    # Edge collision (bit-wise, weighted)
    for u, v, w in edges:
        for b in range(M):
            qu, qv = u*M+b, v*M+b
            key = (min(qu,qv), max(qu,qv))
            J[key] = J.get(key, 0) + w / 2.0

    # Invalid-state penalty  (x0*x1 = (1-Z0)(1-Z1)/4)
    A = INVALID_PENALTY
    for i in range(N):
        q0, q1 = i*M, i*M+1
        h[q0] = h.get(q0, 0) - A/4.0
        h[q1] = h.get(q1, 0) - A/4.0
        key = (q0, q1)
        J[key] = J.get(key, 0) + A/4.0

    return h, J

# ── QAOA Circuit ────────────────────────────────────────────────
def build_circuit(num_q, h, J, gamma, beta, p):
    qc = QuantumCircuit(num_q)
    qc.h(range(num_q))
    for lay in range(p):
        for q, c in h.items():
            qc.rz(2*gamma[lay]*c, q)
        for (q1,q2), c in J.items():
            qc.cx(q1, q2)
            qc.rz(2*gamma[lay]*c, q2)
            qc.cx(q1, q2)
        for q in range(num_q):
            qc.rx(2*beta[lay], q)
    return qc

# ── Classical brute-force optimal ───────────────────────────────
def brute_force_optimal(N, edges):
    best, best_colors = 0, []
    for s in range(K**N):
        cols, t = [], s
        for _ in range(N):
            cols.append(t % K); t //= K
        sc = sum(w for u,v,w in edges if cols[u]!=cols[v])
        if sc > best:
            best, best_colors = sc, [tuple(cols)]
        elif sc == best:
            best_colors.append(tuple(cols))
    return best, best_colors

# ── Evaluation helpers ──────────────────────────────────────────
def decode(bs, N):
    """Decode reversed-bitstring to colors list."""
    colors = []
    for i in range(N):
        bits = bs[i*M:(i+1)*M]
        colors.append(int(bits, 2))
    return colors

def ising_energy(bs, h, J):
    z = [1 if b=='0' else -1 for b in bs]
    e = sum(c*z[q] for q,c in h.items())
    e += sum(c*z[q1]*z[q2] for (q1,q2),c in J.items())
    return e

# ── Transpile metrics ──────────────────────────────────────────
PM = generate_preset_pass_manager(
    optimization_level=3,
    basis_gates=['cx','id','rz','sx','x']
)

def transpile_metrics(qc):
    t = PM.run(qc)
    ops = t.count_ops()
    return t.depth(), ops.get('cx',0), t.size()

# ── Run one topology ───────────────────────────────────────────
def run_one(name, N, edges, p=P_LAYERS):
    num_q = N * M
    h, J = build_liu_ising(N, edges)
    opt_score, opt_cols = brute_force_optimal(N, edges)
    total_w = sum(w for _,_,w in edges)

    best_cost, best_x = float('inf'), None
    for seed in range(NUM_SEEDS):
        rng = np.random.RandomState(seed*42+7)
        x0 = rng.uniform(-np.pi, np.pi, 2*p)
        def cost(params):
            g, b = params[:p], params[p:]
            qc = build_circuit(num_q, h, J, g, b, p)
            sv = Statevector(qc)
            probs = sv.probabilities_dict()
            return sum(prob * ising_energy(bs[::-1], h, J)
                       for bs, prob in probs.items())
        res = minimize(cost, x0, method='L-BFGS-B',
                       options={'maxiter': 500})
        if res.fun < best_cost:
            best_cost, best_x = res.fun, res.x

    # Analyse final state
    qc = build_circuit(num_q, h, J, best_x[:p], best_x[p:], p)
    sv = Statevector(qc)
    probs = sv.probabilities_dict()

    valid_p, opt_p, best_found = 0, 0, 0
    for bs, prob in probs.items():
        cols = decode(bs[::-1], N)
        valid = all(c < K for c in cols)
        if valid:
            valid_p += prob
            sc = sum(w for u,v,w in edges if cols[u]!=cols[v])
            best_found = max(best_found, sc)
            if sc == opt_score:
                opt_p += prob

    # Transpile metrics (bind random params)
    angles = np.random.uniform(-np.pi, np.pi, 2*p)
    qc_bound = build_circuit(num_q, h, J, angles[:p], angles[p:], p)
    depth, cx, gates = transpile_metrics(qc_bound)

    return {
        'name': name, 'nq': num_q,
        'opt': opt_score, 'found': best_found, 'total_w': total_w,
        'valid': valid_p*100, 'optimal': opt_p*100,
        'invalid': (1-valid_p)*100,
        'depth': depth, 'cx': cx, 'gates': gates,
    }

# ── Main ────────────────────────────────────────────────────────
def main():
    print("="*120)
    print(f" Liu et al. Binary QAOA (p={P_LAYERS}) — Weighted MWCSP Benchmark")
    print(f" Encoding: Binary (m={M} bits/node) | K={K} | Invalid penalty={INVALID_PENALTY}")
    print(f" Simulation: Exact Statevector | Optimizer: L-BFGS-B ({NUM_SEEDS} seeds)")
    print("="*120)
    hdr = (f"{'Topology':<20}|{'Qb':>4}|{'Opt':>5}|{'Found':>6}|"
           f"{'Valid%':>8}|{'Opt%':>8}|{'Inv%':>8}|"
           f"{'Depth':>7}|{'CX':>6}|{'Gates':>7}")
    print(hdr)
    print("-"*120)
    for name, d in TOPOS.items():
        t0 = time.time()
        r = run_one(name, d['N'], d['edges'])
        dt = time.time()-t0
        print(f"{r['name']:<20}|{r['nq']:>4}|{r['opt']:>5}|{r['found']:>6}|"
              f"{r['valid']:>7.2f}%|{r['optimal']:>7.2f}%|{r['invalid']:>7.2f}%|"
              f"{r['depth']:>7}|{r['cx']:>6}|{r['gates']:>7}  ({dt:.1f}s)")
    print("="*120)

if __name__ == "__main__":
    main()
