import os
import math
import json
import time
import uuid
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass

# Qiskit core
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.synthesis.evolution import LieTrotter

# Aer noise simulator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# IBM Runtime
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeFez

# Plotting & Math
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

# Settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['figure.dpi'] = 140
matplotlib.rcParams['savefig.dpi'] = 300

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)
np.random.seed(42)

# Data Structures

@dataclass(frozen=True)
class PauliTermInt:
    coeff: complex
    x_mask: int
    y_mask: int
    z_mask: int

    @property
    def flip_mask(self) -> int:
        return self.x_mask | self.y_mask

    @property
    def nY(self) -> int:
        return self.y_mask.bit_count()

    def apply(self, ket: int) -> Tuple[int, complex]:
        bra = ket ^ self.flip_mask
        ones_y = (ket & self.y_mask).bit_count()
        ones_z = (ket & self.z_mask).bit_count()
        phase = (1j ** self.nY) * ((-1) ** (ones_y + ones_z))
        return bra, self.coeff * phase

class PauliHamiltonianInt:
    def __init__(self, n_qubits: int, terms: List[PauliTermInt]):
        self.n_qubits = n_qubits
        self.terms = terms

    @property
    def dim(self) -> int:
        return 1 << self.n_qubits

    def matrix_element(self, bra: int, ket: int) -> complex:
        acc = 0.0 + 0.0j
        for t in self.terms:
            b, amp = t.apply(ket)
            if b == bra:
                acc += amp
        return acc

    def diagonal(self, state: int) -> float:
        val = self.matrix_element(state, state)
        return float(np.real_if_close(val, tol=1e-12))

    def connected(self, state: int) -> List[int]:
        conn = set()
        for t in self.terms:
            b, amp = t.apply(state)
            if abs(amp) > 1e-15:
                conn.add(b)
        return list(conn)

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        labels = []
        coeffs = []
        n = self.n_qubits
        for term in self.terms:
            s = ["I"] * n
            for q in range(n):
                bit = 1 << q
                if term.x_mask & bit: s[n - 1 - q] = "X"
                elif term.y_mask & bit: s[n - 1 - q] = "Y"
                elif term.z_mask & bit: s[n - 1 - q] = "Z"
            labels.append("".join(s))
            coeffs.append(term.coeff)
        return SparsePauliOp(labels, coeffs=np.array(coeffs, dtype=complex))

# Model Builders

def make_heisenberg(n: int, J: float, h_disorder: float, seed: int, pbc: bool = True) -> PauliHamiltonianInt:
    rng = np.random.default_rng(seed)
    terms = []
    _mask = lambda q: 1 << q
    for i in range(n):
        j = (i + 1) % n
        if (not pbc) and (i == n - 1): continue
        terms.append(PauliTermInt(J, _mask(i)|_mask(j), 0, 0))
        terms.append(PauliTermInt(J, 0, _mask(i)|_mask(j), 0))
        terms.append(PauliTermInt(J, 0, 0, _mask(i)|_mask(j)))
        hi = float(h_disorder * rng.standard_normal())
        terms.append(PauliTermInt(hi, 0, 0, _mask(i)))
    return PauliHamiltonianInt(n, terms)

def make_tfim(n: int, J: float, hx: float, hz_disorder: float, seed: int, pbc: bool = True) -> PauliHamiltonianInt:
    rng = np.random.default_rng(seed)
    terms = []
    _mask = lambda q: 1 << q
    for i in range(n):
        j = (i + 1) % n
        if (not pbc) and (i == n - 1): continue
        terms.append(PauliTermInt(-J, 0, 0, _mask(i)|_mask(j)))
    for i in range(n):
        terms.append(PauliTermInt(-hx, _mask(i), 0, 0))
        gi = float(hz_disorder * rng.standard_normal())
        terms.append(PauliTermInt(gi, 0, 0, _mask(i)))
    return PauliHamiltonianInt(n, terms)

def build_model(model_cfg: Dict[str, Any], n: int, seed: int) -> PauliHamiltonianInt:
    name = model_cfg["name"].lower()
    pbc = bool(model_cfg.get("pbc", True))
    if name == "heisenberg":
        return make_heisenberg(n, model_cfg.get("J", 1.0), model_cfg.get("h_disorder", 0.5), seed, pbc)
    if name == "tfim":
        return make_tfim(n, model_cfg.get("J", 1.0), model_cfg.get("hx", 1.0), model_cfg.get("hz_disorder", 0.5), seed, pbc)
    raise ValueError(f"Unknown model: {name}")

# Core

def exact_lowest_2(H: PauliHamiltonianInt):
    dim = H.dim
    rows, cols, data = [], [], []
    for ket in range(dim):
        for term in H.terms:
            bra, amp = term.apply(ket)
            if abs(amp) < 1e-15: continue
            rows.append(bra); cols.append(ket); data.append(amp)
    Hs = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=complex).tocsr()
    Hs = (Hs + Hs.getH()) * 0.5
    vals, vecs = eigsh(Hs, k=2, which="SA")
    idx = np.argsort(vals)
    return vals[idx[0]], vals[idx[1]], vecs[:, idx[0]], vecs[:, idx[1]]

@dataclass
class SampleResult:
    counts: Dict[int, int]
    meta: Dict[str, Any]
    ground_energy: float
    first_energy: Optional[float] = None

def build_state_prep_circuit(H: PauliHamiltonianInt, state_cfg: Dict[str, Any], time_param: float) -> QuantumCircuit:
    n = H.n_qubits
    qc = QuantumCircuit(n)
    init = state_cfg.get("init", "plus").lower()
    if init == "plus": qc.h(range(n))
    op = H.to_sparse_pauli_op()
    reps = int(state_cfg.get("trotter_reps", 1))
    gate = PauliEvolutionGate(op, time=time_param, synthesis=LieTrotter(reps=reps))
    qc.append(gate, qargs=list(range(n)))
    qc.measure_all()
    return qc

def get_samples(H: PauliHamiltonianInt, cfg: Dict[str, Any], seed: int) -> SampleResult:
    mode = cfg["mode"].lower()
    shots = int(cfg["shots"])
    contamination = float(cfg.get("contamination", 0.2))
    e0, e1, psi0, psi1 = exact_lowest_2(H)

    if mode == "exact":
        probs = (1 - contamination) * (np.abs(psi0) ** 2) + contamination * (np.abs(psi1) ** 2)
        probs /= probs.sum()
        draws = np.random.default_rng(seed).multinomial(shots, probs)
        counts = {i: int(c) for i, c in enumerate(draws) if c > 0}
        return SampleResult(counts, {"mode": "exact"}, e0, e1)

    elif mode == "aer_noise":
        sim = AerSimulator(noise_model=NoiseModel.from_backend(FakeFez()))
        state_cfg = cfg["state_prep"]
        qc0 = build_state_prep_circuit(H, state_cfg, state_cfg.get("t0", 0.6))
        qc1 = build_state_prep_circuit(H, state_cfg, state_cfg.get("t1", 1.2))
        pm = generate_preset_pass_manager(backend=sim, optimization_level=1)
        
        c0 = sim.run(pm.run(qc0), shots=int(shots*(1-contamination)), seed_simulator=seed).result().get_counts()
        c1 = sim.run(pm.run(qc1), shots=int(shots*contamination), seed_simulator=seed+1).result().get_counts()
        
        counts = {}
        for c in [c0, c1]:
            for k, v in c.items():
                ikey = int(k.replace(" ", ""), 2)
                counts[ikey] = counts.get(ikey, 0) + v
        return SampleResult(counts, {"mode": "aer_noise"}, e0, e1)

    raise ValueError(f"Mode {mode} not fully implemented in this script version.")

# --- SQD Solver ---

def build_restricted_matrix(H: PauliHamiltonianInt, basis: List[int]):
    m = len(basis)
    index = {b: i for i, b in enumerate(basis)}
    rows, cols, data = [], [], []
    for c, ket in enumerate(basis):
        for term in H.terms:
            bra, amp = term.apply(ket)
            r = index.get(bra)
            if r is not None and abs(amp) > 1e-15:
                rows.append(r); cols.append(c); data.append(amp)
    Hs = coo_matrix((data, (rows, cols)), shape=(m, m), dtype=complex).tocsr()
    return (Hs + Hs.getH()) * 0.5

@dataclass
class SQDState:
    S: Set[int]; E: float; coeffs: np.ndarray; basis: List[int]

@dataclass
class ExpansionParams:
    B_add: int; T_steps: int; tau_dom: float; eps_denom: float; hop: int

class ExpansionSQD:
    def __init__(self, H: PauliHamiltonianInt, counts: Dict[int, int], K_init: int, params: ExpansionParams):
        self.H = H
        self.params = params
        top_k = [b for b, _ in sorted(counts.items(), key=lambda x: -x[1])[:K_init]]
        self.S = set(top_k)

    def _solve_current(self):
        basis = sorted(self.S)
        Hs = build_restricted_matrix(self.H, basis)
        vals, vecs = eigsh(Hs, k=1, which="SA")
        return SQDState(self.S, float(vals[0]), vecs[:, 0], basis)

    def run(self, method: str) -> SQDState:
        for _ in range(self.params.T_steps):
            st = self._solve_current()
            if method == "standard": break
            
            # Simplified Expansion Logic
            probs = np.abs(st.coeffs)**2
            dom_idx = np.where(probs > self.params.tau_dom)[0]
            dom_basis = [st.basis[i] for i in dom_idx]
            
            candidates = set()
            for s in dom_basis:
                for k in self.H.connected(s):
                    if k not in self.S: candidates.add(k)
            
            if not candidates: break
            
            cand_list = list(candidates)
            # Scoring: Energy-based (en)
            scores = []
            for k in cand_list:
                Hkk = self.H.diagonal(k)
                # Approximation of coupling
                nu = sum(st.coeffs[i] * self.H.matrix_element(k, st.basis[i]) for i in range(len(st.basis)))
                scores.append(abs(nu)**2 / max(abs(st.E - Hkk), self.params.eps_denom))
            
            top_adds = np.argsort(scores)[-self.params.B_add:]
            for i in top_adds: self.S.add(cand_list[i])
            
        return self._solve_current()

# Plotting

def plot_results(df: pd.DataFrame, run_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in df['method'].unique():
        sub = df[df['method'] == method]
        ax.plot(sub['n_qubits'], sub['abs_err'], marker='o', label=method)
    ax.set_yscale('log')
    ax.set_xlabel('Qubits'); ax.set_ylabel('Abs Error'); ax.legend()
    plt.title(f"Benchmark: {run_name}")
    plt.savefig(f"figures/{run_name}.png")
    print(f"Plot saved to figures/{run_name}.png")

# Main Execution

if __name__ == "__main__":
    # Config: Exact Mode Example
    CFG_EXACT = {
        "run_name": "exact_demo",
        "mode": "exact",
        "shots": 5000,
        "system_sizes": [8, 10, 12],
        "seeds": [0],
        "models": [{"name": "heisenberg", "J": 1.0, "h_disorder": 0.5}],
        "solver": {"K_init": 40, "B_add": 10, "T_steps": 5, "tau_dom": 1e-4, "eps_denom": 1e-6, "hop": 1},
        "methods": ["standard", "en"]
    }

    CFG = CFG_EXACT
    
    # Account setup (Optional)
    try:
        service = QiskitRuntimeService(channel="ibm_cloud", token="YOUR_TOKEN", instance="YOUR_INSTANCE")
    except:
        print("Runtime Service not configured, proceeding with Local/Exact modes.")

    results = []
    for model_cfg in CFG["models"]:
        for n in CFG["system_sizes"]:
            H = build_model(model_cfg, n, seed=42)
            sample_res = get_samples(H, CFG, seed=123)
            
            params = ExpansionParams(**{k: CFG["solver"][k] for k in ["B_add", "T_steps", "tau_dom", "eps_denom", "hop"]})
            
            for method in CFG["methods"]:
                solver = ExpansionSQD(H, sample_res.counts, CFG["solver"]["K_init"], params)
                st = solver.run(method)
                results.append({
                    "n_qubits": n, "method": method, 
                    "E_est": st.E, "E_true": sample_res.ground_energy,
                    "abs_err": abs(st.E - sample_res.ground_energy)
                })

    df = pd.DataFrame(results)
    df.to_csv(f"data/{CFG['run_name']}.csv", index=False)
    print(df)
    plot_results(df, CFG['run_name'])
