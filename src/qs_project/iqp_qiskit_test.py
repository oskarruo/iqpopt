import numpy as np
from iqp_to_qiskit import IqpCircuitQiskit

def hardware_efficient_iqp_gates(n_qubits):
    """
    Nearest-neighbour IQP ansatz.
    
    Returns:
        gates: list of gates in IQP format [[[i,j]], ...]
    """
    gates = []

    # Single-qubit Z terms
    for i in range(n_qubits):
        gates.append([[i]])

    # Nearest-neighbour ZZ interactions
    for i in range(n_qubits - 1):
        gates.append([[i, i+1]])

    return gates

# Test with hardware efficient gates
if __name__ == "__main__":
    n_qubits = 4
    gates = hardware_efficient_iqp_gates(n_qubits)
    iqp = IqpCircuitQiskit(n_qubits=n_qubits, gates=gates)
    
    params = np.random.rand(len(gates)) * np.pi
    samples = iqp.sample(params, shots=1000)
    probs = iqp.probs(params)

    ops = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [1,1,0,0],
        [0,0,1,1],
    ])
    exp_vals = iqp.op_expval(params, ops, shots=1000)

    print("Samples:", samples)
    print("Probs:", probs)
    print("Expectation values:")
    for op, val in zip(ops, exp_vals):
        print(f"{op} -> {val}")