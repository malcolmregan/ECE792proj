from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info.operators import Operator
from qiskit import Aer
import numpy as np
from math import pi
from scipy.linalg import expm
from qiskit.aqua.algorithms.single_sample import HHL
from qiskit.aqua import run_algorithm
from qiskit.aqua.input import LinearSystemInput
from qiskit.aqua import QuantumInstance

# LANL Example

A = np.asarray([[0.75, 0.25],\
                [0.25, 0.75]])
b = np.asarray([2,0])

'''
# From the paper, 'Quantum Circuit Design for Solving 
# Linear Systems of Equations'
A = 0.25*np.asarray([[15, 9, 5, -3],\
                     [9, 15, 3, -5],\
                     [5, 3, 15, -9],\
                     [-3, -5, -9, 15]])
b = 0.5*np.asarray([1,1,1,1])
'''

params = {
    'problem': {
        'name': 'linear_system'
    },
    'algorithm': {
        'name': 'HHL'
    },
    'eigs': {
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'name': 'EigsQPE',
        'num_ancillae': 3,
        'num_time_slices': 50
    },
    'reciprocal': {
        'name': 'Lookup'
    },
    'backend': {
        'provider': 'qiskit.BasicAer',
        'name': 'statevector_simulator'
    }
}


params['algorithm'] = {
    'truncate_powerdim': False,
    'truncate_hermitian': False
}
params['reciprocal'] = {
    'name': 'Lookup',
    'negative_evals': True
}
params['eigs'] = {
    'expansion_mode': 'suzuki',
    'expansion_order': 2,
    'name': 'EigsQPE',
    'negative_evals': True,
    'num_ancillae': 2,#6
    'num_time_slices': 2#70
}
params['initial_state'] = {
    'name': 'CUSTOM'
}
params['iqft'] = {
    'name': 'STANDARD'
}
params['qft'] = {
    'name': 'STANDARD'
}

algo_input = LinearSystemInput(matrix=A, vector=b)
hhl = HHL.init_params(params, algo_input)
circ = hhl.construct_circuit()
print(circ)

backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend=backend)
result = hhl.run(quantum_instance)
for key in result.keys():
    print(key, result[key])
print("solution ", np.round(result['solution'], 5))
    
