import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from collections import Counter
import sys

# 1. Pauli & Hamiltonian Logic (Physics Definition)
class PauliTerm:
    def __init__(self, coeff, ops):
        self.coeff = coeff
        self.ops = ops
    def apply_to_basis(self, state_int):
        new_state = state_int
        phase = 1.0
        for q_idx, op in self.ops:
            bit = (state_int >> q_idx) & 1
            if op == 'X':
                new_state ^= (1 << q_idx)
            elif op == 'Y':
                new_state ^= (1 << q_idx)
                phase *= 1j if bit == 0 else -1j
            elif op == 'Z':
                if bit == 1: phase *= -1.0
        return new_state, phase * self.coeff
class PauliHamiltonian:
    def __init__(self, n_qubits, terms):
        self.n_qubits = n_qubits
        self.terms = terms 
    def get_matrix_element(self, bra, ket):
        elem = 0.0 + 0.0j
        for term in self.terms:
            target, val = term.apply_to_basis(ket)
            if target == bra:
                elem += val
        return elem.real
    def get_diagonal(self, state):
        return self.get_matrix_element(state, state)
    def get_connected_states(self, state):
        connected = set()
        for term in self.terms:
            target, val = term.apply_to_basis(state)
            if abs(val) > 1e-12:
                connected.add(target)
        return connected
def make_heisenberg_model(n_qubits, J=1.0, h=0.5, seed=42):
    terms = []
    rng = np.random.default_rng(seed)
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        terms.append(PauliTerm(J, [(i, 'X'), (j, 'X')]))
        terms.append(PauliTerm(J, [(i, 'Y'), (j, 'Y')]))
        terms.append(PauliTerm(J, [(i, 'Z'), (j, 'Z')]))
        h_val = h * rng.standard_normal()
        terms.append(PauliTerm(h_val, [(i, 'Z')]))
    return PauliHamiltonian(n_qubits, terms)
# 2. Quantum Device Simulation (Imperfect Preparation)
class NoisyQuantumDevice:
    def __init__(self, hamiltonian, contamination=0.2):
        self.H_def = hamiltonian
        self.dim = 1 << hamiltonian.n_qubits
        self.contamination = contamination
        self._solve_ground_truth_and_noise()
    def _solve_ground_truth_and_noise(self):
# Build Sparse Matrix for simulation (not exposed to solver)
        row, col, data = [], [], []
# Efficient sparse build for simulation (N <= 12 is fast)
        for i in range(self.dim):
            for term in self.H_def.terms:
                target, val = term.apply_to_basis(i)
                row.append(target)
                col.append(i)
                data.append(val.real)
        H_sparse = coo_matrix((data, (row, col)), shape=(self.dim, self.dim)).tocsr()
# Get lowest 2 eigenvalues
# k=2 to simulate excited state contamination
        evals, evecs = eigsh(H_sparse, k=2, which='SA')
        self.ground_energy = evals[0]
        psi0 = evecs[:, 0]
        psi1 = evecs[:, 1]
# Simulating Imperfect State Preparation (e.g. mediocre VQE)
# Probability = (1-p)|psi0|^2 + p|psi1|^2
        self.probs = (1 - self.contamination) * np.abs(psi0)**2 + \
                     self.contamination * np.abs(psi1)**2
# Ensure sum to 1
        self.probs /= np.sum(self.probs)
    def measure_shots(self, n_shots, seed=None):
        rng = np.random.default_rng(seed)
        counts_list = rng.multinomial(n_shots, self.probs)
        results = {idx: count for idx, count in enumerate(counts_list) if count > 0}
        return results
# 3. Solvers
class BaseSQD:
    def __init__(self, hamiltonian, shot_counts):
        self.H_def = hamiltonian
# Initial basis from shots
        sorted_counts = sorted(shot_counts.items(), key=lambda x: -x[1])
# Take top K shots as starting point (e.g., top 50)
        self.S = set(k for k, v in sorted_counts[:50])
    def solve_subspace(self):
        basis_list = sorted(list(self.S))
        n = len(basis_list)
        if n == 0: return 0.0, [], []
        H_sub = np.zeros((n, n))
        for r, bra in enumerate(basis_list):
            for c, ket in enumerate(basis_list):
                if c < r: continue
                val = self.H_def.get_matrix_element(bra, ket)
                H_sub[r, c] = H_sub[c, r] = val
        evals, evecs = eigh(H_sub)
        return evals[0], evecs[:, 0], basis_list
class StandardSQD(BaseSQD):
    def run(self):
        E0, _, _ = self.solve_subspace()
        return E0
class RandomSQD(BaseSQD):
    def step(self):
        E0, _, basis_list = self.solve_subspace()
# Generate ALL connected candidates
        candidates = set()
        for state in basis_list:
            candidates.update(self.H_def.get_connected_states(state))
        candidates -= self.S
        if not candidates: return E0
# Randomly pick candidates
        import random
        n_add = min(len(candidates), 20)
        chosen = random.sample(list(candidates), n_add)
        self.S.update(chosen)
        return E0
class AS_SQD(BaseSQD):
    def step(self):
        E0, c0, basis_list = self.solve_subspace()
# 1. Generate Candidates
        candidates = set()
# Optimization: Only expand from dominant basis vectors
        dominant_indices = [i for i, c in enumerate(c0) if abs(c)**2 > 1e-5]
        for idx in dominant_indices:
            state = basis_list[idx]
            candidates.update(self.H_def.get_connected_states(state))
        candidates -= self.S
        if not candidates: return E0
        cand_list = list(candidates)
# 2. Score Candidates (Epstein-Nesbet)
# score = |<k|H|psi>|^2 / (E0 - <k|H|k>)
# Note: Denominator (E0 - Hkk) is usually negative.
# We take abs value or handle sign to rank 'importance'.
        scores = []
        H_diag = np.array([self.H_def.get_diagonal(k) for k in cand_list])
# Denom regularization
        denoms = np.abs(E0 - H_diag)
        denoms = np.maximum(denoms, 1e-6) 
        for k_i, k_state in enumerate(cand_list):
# Compute <k|H|psi> = sum_j c_j <k|H|j>
# Only need to sum over j connected to k
            num = 0.0
# To avoid N^2 loop, we reconstruct efficiently:
# But for simplicity in demo, we iterate non-zero c0
            for basis_idx in dominant_indices:
                b_state = basis_list[basis_idx]
                val = self.H_def.get_matrix_element(k_state, b_state)
                if val != 0:
                    num += val * c0[basis_idx]
            scores.append((abs(num)**2) / denoms[k_i])
# 3. Select Top
        n_add = min(len(scores), 20)
        top_indices = np.argsort(scores)[-n_add:]
        for i in top_indices:
            self.S.add(cand_list[i])
        return E0
# 4. Scaling Benchmark Logic
def run_scaling_benchmark():
    SYSTEM_SIZES = [8, 10, 12] # N=12 is approx 4096 states
    N_SHOTS = 2000
    STEPS = 10
    SEEDS = 5
    CONTAMINATION = 0.2 # 20% Excited State mixed in
    results = {
        "Standard": [],
        "Random": [],
        "AS-SQD": []
    }
    for n_q in SYSTEM_SIZES:
        # Storage for this N
        err_std = []
        err_rnd = []
        err_as = []
        for s in range(SEEDS):
# 1. Setup Environment
            H = make_heisenberg_model(n_q, seed=s+100)
            dev = NoisyQuantumDevice(H, contamination=CONTAMINATION)
            shots = dev.measure_shots(N_SHOTS, seed=s+200)
            E_true = dev.ground_energy
# 2. Standard SQD (Subspace only)
            std = StandardSQD(H, shots)
            e_std = std.run()
            err_std.append(abs(e_std - E_true))
# 3. Random SQD (Blind expansion)
            rnd = RandomSQD(H, shots)
            for _ in range(STEPS): rnd.step()
            e_rnd = rnd.solve_subspace()[0]
            err_rnd.append(abs(e_rnd - E_true))
# 4. AS-SQD (Smart expansion)
            as_sqd = AS_SQD(H, shots)
            for _ in range(STEPS): as_sqd.step()
            e_as = as_sqd.solve_subspace()[0]
            err_as.append(abs(e_as - E_true))
            sys.stdout.write(f".")
            sys.stdout.flush() 
# Store Median Errors
        for k, v in zip(("Standard", "Random", "AS-SQD"),(err_std, err_rnd, err_as)):
            results[k].append(np.median(v))

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(SYSTEM_SIZES, results["Standard"], 's--', label='Standard SQD (Sampled Subspace)', color='gray')
    plt.plot(SYSTEM_SIZES, results["Random"], '^-', label='Random Expansion', color='orange')
    plt.plot(SYSTEM_SIZES, results["AS-SQD"], 'o-', label='AS-SQD (Proposed)', color='blue', linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('System Size (Qubits)')
    plt.ylabel('Energy Error |E - E_true| (Hartree)')
    plt.title('Scaling Analysis: Error Recovery from Noisy States')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xticks(SYSTEM_SIZES)
    
    filename = "scaling_benchmark.png"
    plt.savefig(filename)
    print(f"\nSaved plot to {filename}")

if __name__ == "__main__":
    run_scaling_benchmark()
