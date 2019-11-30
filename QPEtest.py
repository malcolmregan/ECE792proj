from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info.operators import Operator
from qiskit import Aer
import numpy as np
from math import pi
from scipy.linalg import expm
from qiskit.aqua.algorithms.single_sample.hhl import hhl
np.set_printoptions(precision=5,linewidth=400)

############################
### function definitions ###
############################

def printcircuitunitary(circ):
    bkend = Aer.get_backend('unitary_simulator')
    job =execute(circ,bkend)
    result = job.result()
    print(result.get_unitary(circ, decimals=3))

def printstatevector(circ):
    bkend = Aer.get_backend('statevector_simulator')
    job = execute(circ, bkend)
    result = job.result()
    print(result.get_statevector())

def getstatevector(circ):
    bkend = Aer.get_backend('statevector_simulator')
    job = execute(circ, bkend)
    result = job.result()
    return result.get_statevector()

def qft(circ, q, n):
    for j in range(n):
        for k in range(j):
            circ.cu1(pi/float(2**(j-k)), q[j], q[k])
        circ.h(q[j])

def iqft(circ, q, n):
    for j in range(n-1,-1,-1):
        circ.h(q[j])
        for k in range(j-1,-1,-1):
            circ.cu1(-pi/float(2**(j-k)), q[k], q[j])

def getUs(clocksize, mat):
    Ulist = list()
    invUlist = list()
    Ulist.append(Operator(mat))
    invUlist.append(Operator(mat.conj().T))
    for i in range(clocksize-1):
        mat = np.matmul(mat,mat)
        Ulist.append(Operator(mat))
        invUlist.append(Operator(mat.conj().T))

    return Ulist, invUlist

def bintoint(string):
    number = 0
    for i in range(len(string)):
        number = number + int(string[len(string)-1-i])*2**i
    return number

def binfractodec(string):
    number = 0
    for i in range(len(string)):
        number = number + int(string[len(string)-1-i])*2**(-i-1)
    return number

def hermtocontU(mat,T):
    # takes uncontrolled matrix
    # must be hermitian currently
    # - add code to make non-hermitian
    # matrix hermitian
    if np.array_equal(mat, mat.conj().T):
        hermop = mat
    expherm = expm(2j*pi*hermop/T)
    
    ''' 
    # one of the unitary operators from Quirk implementation of 4x4 example circuit
    # for comparison with the above operators - exact up to 4 decimal places
    eAitover8 = np.asarray([[0.1767592+0.4267675j,-0.1767709-0.4267956j,\
            0.323234+0.0731928j,-0.6767777+0.0731976j],\
            [-0.1767592-0.4267675j,0.1767709+0.4267956j,\
            0.676799-0.0731928j,-0.3231894-0.0731976j],\
            [0.3232573+0.073249j,0.6767544-0.0732538j,\
            0.1767825+0.4267578j,0.1767942+0.4267859j],\
            [-0.6767757+0.073249j,-0.3232126-0.0732538j,\
            0.1767825+0.4267578j,0.1767942+0.4267859j]])
    '''

    # add control
    M0 = np.asarray([[1,0],\
                     [0,0]])
    M1 = np.asarray([[0,0],\
                     [0,1]])
    I = np.eye(np.shape(hermop)[0])
    cexpherm = np.kron(M0,I)+np.kron(M1,expherm)
    
    return cexpherm

def prepareb(vector,circ, qb):
    # add circuit elements to prepare the state
    # the normalized vector as a state
    
    normfactor = np.sqrt(np.sum(vector**2))
    state = vector/normfactor

    circ.initialize(state, [qbtox[i] for i in range(len(qb))])
    circ.barrier()

    return normfactor

####################################################
### problem variables and circuit initialization ###
####################################################

# LANL Example
A = np.asarray([[0.75, 0.25],\
                [0.25, 0.75]])
b = 1/np.sqrt(2)*np.asarray([1,1]) 
w,v = np.linalg.eig(A)
print(w,v)

'''
# From the paper, 'Quantum Circuit Design for Solving 
# Linear Systems of Equations'
A = 0.25*np.asarray([[15, 9, 5, -3],\
                     [9, 15, 3, -5],\
                     [5, 3, 15, -9],\
                     [-3, -5, -9, 15]])
b = 0.5*np.asarray([1,1,1,1])
'''

T = 2

cexpherm = hermtocontU(A,T)
clocksize = 2

# qbtox size wont work if b and cepherm dimensions are not a power of 2
# need to modify hermtocontU() and prepareb() to make sure they produce 
# stuff with dimension 2^n by 2^n. this may change how the result is to be
# interpreted, so hermtocontU() should also return flags indicating
# how to proceed in later parts of the algorithm

qclock = QuantumRegister(clocksize,'clk')
qbtox = QuantumRegister(np.log2(np.shape(b)[0]),'btox')
cclock = ClassicalRegister(clocksize,'cclk')
cbtox = ClassicalRegister(np.log2(np.shape(b)[0]),'cbtox')

circ = QuantumCircuit(qbtox,qclock,cbtox,cclock)

bnormfactor = prepareb(b,circ,qbtox)

################################
### quantum phase estimation ###
################################

# should quantum phase estimation be run
# first and then eigenvalues measured 
# to get the condition number
# not efficient but would be better for 
# understanding/explaining how things work

circ.h(qclock)
circ.barrier()

Ulist,_ = getUs(len(qclock), cexpherm)
for i in range(len(Ulist)):
    reglist = [qbtox[k] for k in range(len(qbtox))]
    reglist.append(qclock[i])
    circ.unitary(Ulist[i], reglist)
circ.barrier()

iqft(circ, qclock, len(qclock))
circ.barrier()

#########################################################
### get statevector for qbtox conditioned on qanc = 1 ###
#########################################################

print('\n############################')
print('### Statevector analysis ###')
print('############################\n')

statevec = getstatevector(circ)
statevec = statevec.reshape(len(statevec),1)
binlen = (len(qclock)+len(qbtox))
zeros='0'*binlen

for i in range(len(statevec)):
    binary = str(bin(i))[2:]
    if len(binary)<binlen:
        binary = zeros[0:-len(binary)]+binary
    print(binary, statevec[i])

print(circ)
