import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import StatePreparation, UnitaryGate, RZGate

# 1. Build 4x4 Space-Time Matrix A (1-country rain, Nt=1 step)
mu = -np.log(1 - 0.02)
p = 1.0 * mu
M = np.array([[1 - p, 0.0], [p, 1.0]])
A = np.block([[np.eye(2), np.zeros((2, 2))], [-M, np.eye(2)]])

# 2. Decompose A into Pauli Strings (SVD-Free LCU)
pauli_op = SparsePauliOp.from_operator(A)
terms = [(lbl, c) for lbl, c in zip(pauli_op.paulis.to_labels(), pauli_op.coeffs) if abs(c) > 1e-12]
L = len(terms)
n_lcu = int(np.ceil(np.log2(L)))
M_anc = 2**n_lcu

c_list = np.zeros(M_anc)
U_list = [np.eye(4, dtype=complex) for _ in range(M_anc)]
for i, (lbl, coeff) in enumerate(terms):
    c_list[i] = abs(coeff)
    U_list[i] = (coeff / abs(coeff)) * SparsePauliOp(lbl).to_matrix()

alpha_lcu = np.sum(c_list)

# 3. Build PREPARE and SELECT Gates
prep = StatePreparation(np.sqrt(c_list / alpha_lcu), label="PREP")
sel_mat = np.zeros((4 * M_anc, 4 * M_anc), dtype=complex)
for l in range(M_anc):
    sel_mat[4*l:4*(l+1), 4*l:4*(l+1)] = U_list[l]
select = UnitaryGate(sel_mat, label="SELECT")

# Assemble U_LCU subcircuit
u_lcu_qc = QuantumCircuit(2 + n_lcu, name="U_LCU")
u_lcu_qc.append(prep, range(2, 2 + n_lcu))
u_lcu_qc.append(select, range(2 + n_lcu))
u_lcu_qc.append(prep.inverse(), range(2, 2 + n_lcu))
U_LCU = u_lcu_qc.to_gate()

# 4. Build Projector-Controlled Phase Gate Π_φ on (lcu_anc == 0, qsvt_anc)
def projector_phase_gate(phi, n_lcu_qubits):
    qc_phi = QuantumCircuit(n_lcu_qubits + 1, name=f"Π({phi:.2f})")
    lcu_q = list(range(n_lcu_qubits))
    sig_q = n_lcu_qubits
    # Flip lcu_anc so |0...0> becomes |1...1>
    qc_phi.x(lcu_q)
    # Multi-controlled Z rotation on qsvt_anc when lcu_anc == |0...0>
    mcrz = RZGate(2 * phi).control(n_lcu_qubits)
    qc_phi.append(mcrz, [*lcu_q, sig_q])
    qc_phi.x(lcu_q)
    return qc_phi.to_gate()

# 5. Assemble Full End-to-End Circuit: State Prep -> QSVT(U_LCU) -> Measurement
state_reg = QuantumRegister(1, "state")      # |r> in {0, 1}
clock_reg = QuantumRegister(1, "clock")      # |t> in {0, 1}
lcu_reg   = QuantumRegister(n_lcu, "lcu")    # 3 LCU ancillas
qsvt_reg  = QuantumRegister(1, "qsvt")       # 1 QSVT signal ancilla
c_out     = ClassicalRegister(6, "meas")

qc = QuantumCircuit(state_reg, clock_reg, lcu_reg, qsvt_reg, c_out)

# Step A: Initial state preparation |b> = |t=0, r=0> (already |00>)
qc.barrier(label="State Prep |b>")

# Step B: QSVT Polynomial Sequence (illustrative degree d=3 odd polynomial phases)
phases = [0.42, -0.85, 0.85, -0.42]
qc.h(qsvt_reg[0])
for k, phi in enumerate(phases):
    qc.append(projector_phase_gate(phi, n_lcu), [*lcu_reg, qsvt_reg[0]])
    if k < len(phases) - 1:
        gate = U_LCU if (k % 2 == 0) else U_LCU.inverse()
        qc.append(gate, [state_reg[0], clock_reg[0], *lcu_reg])
qc.h(qsvt_reg[0])

# Step C: Post-selection & Terminal Payoff Measurement
qc.barrier(label="Post-Select & Readout")
qc.measure([*state_reg, *clock_reg, *lcu_reg, *qsvt_reg], c_out)

print(qc.draw(output="text", fold=120))