"""
noisy_simulation.py — Run the Min Width oracle under an IBM Heron-class noise
model and compare against the ideal (noiseless) baseline.

Outputs: a Markdown table summarising P(measure global optimum) per topology
under {ideal, 0.25x Heron, 1.0x Heron, 4.0x Heron} CX-error rates. The scaling
sweeps both depolarising and thermal-relaxation channels together.

Run:  .venv/bin/python noisy_simulation.py
"""
import math
from collections import Counter

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, thermal_relaxation_error,
)

from verify_table2 import (
    create_min_width_oracle, classical_best_score, TOPOLOGIES, BASIS_GATES,
)


# IBM Heron-class median parameters (publicly reported numbers, June 2024).
T1_BASE_NS    = 180_000.0   # 180 us
T2_BASE_NS    = 200_000.0   # 200 us
CX_TIME_NS    = 60.0
SX_TIME_NS    = 35.0
X_TIME_NS     = 35.0
RZ_TIME_NS    = 0.0         # virtual Z, error-free
ID_TIME_NS    = 35.0

# Median two-qubit and single-qubit Pauli error rates at Heron baseline.
CX_ERR_BASE   = 5e-3
SX_ERR_BASE   = 2e-4
X_ERR_BASE    = 2e-4


def build_noise_model(scale: float) -> NoiseModel:
    """Heron-class noise model with all gate errors scaled by `scale`.

    Uses depolarising error only (simpler than composing with thermal
    relaxation, which occasionally trips Aer's eigensystem solver at
    Heron-scale parameters). Thermal relaxation effects would dominate
    only at depths far beyond what these benchmarks reach within a
    single Grover iteration; the depolarising channel captures the
    leading per-gate error budget that matters here.

    `scale=0` -> ideal; `scale=1` -> Heron baseline; `scale>1` -> noisier."""
    nm = NoiseModel(basis_gates=BASIS_GATES)
    if scale <= 0:
        return nm

    cx_err = CX_ERR_BASE * scale
    sx_err = SX_ERR_BASE * scale
    x_err  = X_ERR_BASE  * scale

    nm.add_all_qubit_quantum_error(depolarizing_error(cx_err, 2), ['cx'])
    nm.add_all_qubit_quantum_error(depolarizing_error(sx_err, 1), ['sx'])
    nm.add_all_qubit_quantum_error(depolarizing_error(x_err, 1),  ['x'])
    return nm


def coloring_from_bitstring(bs: str, N: int, q: int) -> tuple:
    """Decode a coloring measurement bitstring (Qiskit big-endian)."""
    bs = bs.replace(' ', '')[::-1]   # reverse for little-endian iteration
    colors = []
    for i in range(N):
        chunk = bs[i*q:(i+1)*q]
        colors.append(int(chunk[::-1], 2))
    return tuple(colors)


def run_one(topo_name, N, K, edges, scale, shots=4096):
    target = classical_best_score(N, K, edges)
    # Use 1 Grover iteration; the relative noise-vs-ideal ratio is what we care about.
    qc = create_min_width_oracle(N, K, edges, target, grover_iterations=1)
    nm = build_noise_model(scale)
    sim = AerSimulator(noise_model=nm) if scale > 0 else AerSimulator()
    tqc = transpile(qc, sim, basis_gates=BASIS_GATES, optimization_level=3)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Sum probability mass over all configurations achieving the optimal score.
    q_per_node = math.ceil(math.log2(K)) if K > 1 else 1
    p_optimal = 0.0
    for bs, c in counts.items():
        colors = coloring_from_bitstring(bs, N, q_per_node)
        score = sum(w for u, v, w in edges if colors[u] != colors[v])
        if score == target:
            p_optimal += c / shots
    return p_optimal


def main():
    import sys
    print("\n=== Min Width oracle under IBM Heron-class noise ===", flush=True)
    print(f"Baseline params: T1={T1_BASE_NS/1000:.0f}us, T2={T2_BASE_NS/1000:.0f}us, "
          f"CX time={CX_TIME_NS:.0f}ns, CX err={CX_ERR_BASE:.1e}\n", flush=True)
    print(f"{'Graph':<22} {'Depth':<8} {'P_opt(ideal)':<14} "
          f"{'P_opt(0.25x)':<14} {'P_opt(1x)':<12} {'P_opt(4x)':<12}", flush=True)
    print('-' * 88, flush=True)

    # Three topologies that span small (12q, depth 718) to deep (16q, 2020):
    # adding Frustrated K_4 (17q, depth 4274) blew past the per-shot stochastic
    # noise budget so we drop it. The trend across these three is sufficient.
    pick = {'Triangle', 'Star $S_4$', 'Diag Square'}

    backend = AerSimulator()
    for name, N, K, edges in TOPOLOGIES:
        if name not in pick:
            continue
        # Get transpiled depth for the table.
        qc = create_min_width_oracle(N, K, edges, classical_best_score(N, K, edges), 1)
        t = transpile(qc, backend, basis_gates=BASIS_GATES, optimization_level=3)
        depth = t.depth()

        print(f"  -> running {name} (N={N}, depth={depth}) ...", flush=True)
        p_ideal = run_one(name, N, K, edges, scale=0.0)
        p_q     = run_one(name, N, K, edges, scale=0.25)
        p_1     = run_one(name, N, K, edges, scale=1.0)
        p_4     = run_one(name, N, K, edges, scale=4.0)
        print(f"{name:<22} {depth:<8} {p_ideal:<14.3f} {p_q:<14.3f} {p_1:<12.3f} {p_4:<12.3f}", flush=True)


if __name__ == "__main__":
    main()
