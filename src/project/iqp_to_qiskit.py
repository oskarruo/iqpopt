from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
import numpy as np

# Note: The class uses a simulator (StatevectorSampler) at the moment but
# the circuit can be used for a real hardware, too. However, the sample function must be modified. 
# params is a list/array of angles, one per gate.
# gates is a list of which qubits each gate acts on.

class IqpCircuitQiskit:
    def __init__(self, n_qubits, gates, init_gates=None, spin_sym=False, bitflip=False):
        self.n_qubits = n_qubits
        self.gates = gates
        self.init_gates = init_gates
        self.spin_sym = spin_sym
        self.bitflip = bitflip
    
    def iqp_circuit(self, params, init_coefs=None):
        qc = QuantumCircuit(self.n_qubits)

        # Spin symmetry
        if self.spin_sym:
            qc.h(0)
            for i in range(1, self.n_qubits):
                qc.cx(0, i)

        # Hadamards
        for i in range(self.n_qubits):
            qc.h(i)

        # ZZ decomposition
        def apply_zz(i, j, theta):
            qc.cx(i, j)
            qc.rz(2 * theta, j)
            qc.cx(i, j)

        # Initial gates
        if self.init_gates is not None:
            for par, gate in zip(init_coefs, self.init_gates):
                qubits = gate[0]

                if len(qubits) == 1:
                    qc.rz(2 * par, qubits[0])

                elif len(qubits) == 2:
                    apply_zz(qubits[0], qubits[1], par)

                else:
                    raise ValueError("Hardware-efficient version supports only 1- and 2-qubit gates")

        # Trainable gates
        # Loop over each gate in the ansatz and its corresponding parameter
        for par, gate in zip(params, self.gates):
            qubits = gate[0]

            if len(qubits) == 1:
                qc.rz(2 * par, qubits[0])

            elif len(qubits) == 2:
                apply_zz(qubits[0], qubits[1], par)

            else:
                raise ValueError("Hardware-efficient version supports only 1- and 2-qubit gates")

        # Final Hadamards
        for i in range(self.n_qubits):
            qc.h(i)

        return qc
        
    def sample(self, params, init_coefs=None, shots=1024):
        qc = self.iqp_circuit(params, init_coefs)

        qc.measure_all()

        sampler = StatevectorSampler()
        job = sampler.run([qc], shots=shots)
        result = job.result()

        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()

        samples = []
        for bitstring, cnt in counts.items():
            arr = np.array([int(b) for b in reversed(bitstring)])
            samples.extend([arr] * cnt)

        return np.array(samples)

    # Only works for a simulator, NOT for real hardware 
    def probs(self, params, init_coefs=None):
        qc = self.iqp_circuit(params, init_coefs)

        # Use Statevector directly
        state = Statevector.from_instruction(qc)
        probs = np.abs(state.data)**2

        return probs
    
    def op_expval(self, params, ops, shots=1000, init_coefs=None):
        """
        Compute expectation values of Pauli-Z operators using sampling.

        Args:
            params: circuit parameters
            ops: array of shape (n_ops, n_qubits), binary (0/1)
                e.g. [1,0,1] = Z ⊗ I ⊗ Z
            shots: number of samples

        Returns:
            expvals: array of expectation values (length n_ops)
        """

        # 1. Sample from circuit
        samples = self.sample(params, init_coefs, shots=shots)
        # shape: (shots, n_qubits)

        # 2. Ensure ops is 2D
        ops = np.array(ops)
        if ops.ndim == 1:
            ops = ops.reshape(1, -1)

        expvals = []

        # 3. Compute expectation values
        for op in ops:
            # pick relevant qubits
            idx = np.where(op == 1)[0]

            if len(idx) == 0:
                expvals.append(1.0)
                continue

            # compute (-1)^(sum of bits)
            vals = (-1) ** np.sum(samples[:, idx], axis=1)

            expvals.append(np.mean(vals))

        return np.array(expvals)