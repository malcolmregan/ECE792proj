from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info.operators import Operator
from qiskit import Aer
import numpy as np
from math import pi
from scipy.linalg import expm
from qiskit.aqua.algorithms.single_sample.hhl import hhl
import json

np.set_printoptions(precision=5,linewidth=400)

############################
### function definitions ###
############################

def userinput():
    b = 0
    r = 0
    clocksize = 0
    T = 0
    A = input('Enter A, an NxN hermitian matrix as list, or enter nothing to run a default example: ') 
    if A is '':
        print('Running default example')
        A = None
    elif not np.array_equal(np.asarray(A), np.asarray(A).conj().T):
        print('A not hermitian, running default example')
        A = None
    if A is not None:
        A = json.loads(A)
        A = np.asarray(A)
        b = input('Enter b, a vector of length N as list: ')
        b = json.loads(b)
        b = np.asarray(b)
        r = int(input('Enter r, denoting how small eigenvalue-controlled rotations will be (angle proportional to 2^(-r)): '))
        T = int(input('Enter T, such that eigenvalues of A/T are all less than 1: '))
        clocksize = int(input('Enter QPE depth: '))
     
    return A, b, r, T, clocksize

def defaultprograms():
    num = '1'
    num = input('Select example:\n\
            [1] LANL example\n\
            [2] Example from \'Quantum Circuit Design for Solving Linear Systems of Eqns\' paper\n\
            [3] Example with eigenvalues 0.375 and 0.25\n>> ')
    if num == '1':
        # LANL Example
        A = np.asarray([[0.75, 0.25],\
                        [0.25, 0.75]])
        b = np.asarray([2,0])
        T = 2
        clocksize = 2
        r = 4

    if num == '2':
        # From the paper, 'Quantum Circuit Design for Solving
        # Linear Systems of Equations'
        A = 0.25*np.asarray([[15,  9, 5,  -3],\
                             [9,  15,  3, -5],\
                             [5,   3, 15, -9],\
                             [-3, -5, -9, 15]])
        b = 0.5*np.asarray([1,1,1,1])
        T = 16
        clocksize = 4
        r = 5
    
    if num == '3':
        # Example with matrix that doesn't have eigenvalues
        # that are a power of 0.5 but that are an exact
        # sum of low powers of 0.5
        A = 2*np.asarray([[0.375,   0],
                          [0,    0.25]])
        b = np.asarray([1,1])
        T = 2
        clocksize = 4
        r = 4

    return A, b, r, T, clocksize

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

    circ.initialize(state, [qb[i] for i in range(len(qb))])
    circ.barrier()

    return normfactor

####################################################
### problem variables and circuit initialization ###
####################################################

A, b, r, T, clocksize = userinput()

if A is None:
    A, b, r, T, clocksize = defaultprograms()

print('A: \n',A)
print('b: \n',b)
print('T: ',T)
print('QPE Depth: ', clocksize)
print('r: ', r)

actualans=np.matmul(np.linalg.inv(A),np.asarray(b).reshape(len(b),1))
eigval,eigvec = np.linalg.eig(A)
print('Eigenvalues of A/T', np.around(eigval/T,decimals=6).tolist())
cond = max(eigval)/min(eigval)

cexpherm = hermtocontU(A,T)

#######################################
### Run QPE and measure eigenvalues ###
#######################################

qpeclock = QuantumRegister(clocksize,'qpeclk')
qpeb = QuantumRegister(np.log2(np.shape(b)[0]),'qpebtox')
qpecclock = ClassicalRegister(clocksize, 'qpecclk')
qpecb = ClassicalRegister(np.log2(np.shape(b)[0]),'qpecb')
qpecirc = QuantumCircuit(qpeclock, qpeb, qpecclock, qpecb)

bnormfactor = prepareb(b,qpecirc,qpeb)
qpecirc.barrier()

qpecirc.h(qpeclock)
qpecirc.barrier()

Ulist,_ = getUs(len(qpeclock), cexpherm)
for i in range(len(Ulist)):
    reglist = [qpeb[k] for k in range(len(qpeb))]
    reglist.append(qpeclock[i])
    qpecirc.unitary(Ulist[i], reglist)
qpecirc.barrier()

iqft(qpecirc, qpeclock, len(qpeclock))
qpecirc.barrier()

qpecirc.measure(qpeclock,qpecclock)
qpecirc.measure(qpeb, qpecb)

shots = 100000
bkend = Aer.get_backend('qasm_simulator')
job = execute(qpecirc, bkend, shots=shots)
result = job.result()
counts = result.get_counts(qpecirc)

QPEeigvals = list()
for key in counts.keys():
    QPEeigvals.append(key[-len(qpeclock):])
QPEeigvals = np.unique(QPEeigvals)

QPEeigvallist = list()
for v in QPEeigvals:
    QPEeigvallist.append(binfractodec(v))
print('Eigenvalues of A/T from QPE:', QPEeigvallist)

###########
### HHL ###
###########

qclock = QuantumRegister(clocksize,'clk')
qbtox = QuantumRegister(np.log2(np.shape(b)[0]),'btox')
qanc = QuantumRegister(1,'anc')
cclock = ClassicalRegister(clocksize,'cclk')
cbtox = ClassicalRegister(np.log2(np.shape(b)[0]),'cbtox')
canc = ClassicalRegister(1,'canc')

circ = QuantumCircuit(qbtox,qclock,qanc,cbtox,cclock,canc)

bnormfactor = prepareb(b,circ,qbtox)

################################
### Quantum Phase Estimation ###
################################

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

#####################
### rotation part ###
#####################

rotationidxs = list()
for val in QPEeigvals:
    rotationidxs.append([len(qclock)-idx-1 for idx, char in enumerate(val) if char == '1'])
rotationidxs.sort(key=len)    

for idxs in rotationidxs:
    angle = 0
    for i in idxs:
        angle = angle + 2**(-i-1)
    circ.mcry(1/(angle*(2**r)),[qclock[k] for k in idxs],qanc[0],q_ancillae=None)
    # find rotationidxs sublists that consist entirely of elements
    # contained in the current sublist
    # only reverse those that have length that are length(current sublist)-1
    for ridxs in rotationidxs:
        if len(ridxs)==len(idxs)-1:
            if set(ridxs).issubset(set(idxs)):
                angle=0
                for j in ridxs:
                    angle = angle + 2**(-j-1)
                circ.mcry(-1/(angle*(2**r)),[qclock[k] for k in idxs],qanc[0],q_ancillae=None)
circ.barrier()

# fidelity of answer goes up with r
# probability of ancilla = 1 for post selection goes down with r

####################
### Reverse QPE  ###         
####################

qft(circ, qclock, len(qclock))
circ.barrier()

_,invUlist = getUs(len(qclock), cexpherm)
for i in range(len(invUlist)):
    reglist = [qbtox[k] for k in range(len(qbtox))]
    reglist.append(qclock[len(invUlist)-1-i])
    circ.unitary(invUlist[len(invUlist)-1-i], reglist)
circ.barrier()

circ.h(qclock)
circ.barrier()

#########################################################
### get statevector for qbtox conditioned on qanc = 1 ###
#########################################################

print('\n############################')
print('### Statevector analysis ###')
print('############################\n')

statevec = getstatevector(circ)
statevec = statevec.reshape(len(statevec),1)
binlen = (len(qclock)+len(qbtox)+len(qanc))
zeros = '0'*binlen

postselectionprob = 0
postselectedvector = list()
postselectedbinaryidx = list()

#print('Full Statevector:')
for i in range(len(statevec)):
    binary = str(bin(i))[2:]
    if len(binary)<binlen:
        binary = zeros[:-len(binary)]+binary
    #print(binary, statevec[i][0])
    if binary[0]=='1':
        postselectionprob=postselectionprob+\
                statevec[i][0]*statevec[i][0].conj()
        postselectedvector.append(statevec[i][0])
        postselectedbinaryidx.append(binary)
normfactor = np.sqrt(np.sum(np.asarray(postselectedvector)**2))
postselectedvector= postselectedvector/normfactor
#print('\n')

postselectionprob = np.real(postselectionprob)*100
#print('Postselected Statevector (Postselection prob - {:.2f}%):'.format(postselectionprob))
qbtoxstate = list()
qbtoxbinidx = list()
for i in range(len(postselectedvector)):
    #print(postselectedbinaryidx[i][1:],postselectedvector[i])
    if postselectedbinaryidx[i][1:1+len(qclock)]=='0'*len(qclock):
        qbtoxbinidx.append(postselectedbinaryidx[i][-len(qbtox):])
        qbtoxstate.append(postselectedvector[i])
#print('\n')

#print('Solution Statevector:')
#for i in range(len(qbtoxstate)):
    #print(qbtoxbinidx[i],qbtoxstate[i])
#print('\n')

finalstatevector = np.asarray(qbtoxstate).reshape(len(qbtoxstate),1)

#####################################
### measure, analyze measurements ###
#####################################

circ.measure(qanc,canc)
circ.measure(qbtox,cbtox)
circ.measure(qclock,cclock)

print('\n###############')
print('### Circuit ###')
print('###############\n')
print(circ)

shots = 100000
bkend = Aer.get_backend('qasm_simulator')
job = execute(circ, bkend, shots=shots)
result = job.result()
counts = result.get_counts(circ)

print('\n############################')
print('### Measurement analysis ###')
print('############################\n')

shots = 100000
bkend = Aer.get_backend('qasm_simulator')
job = execute(circ, bkend, shots=shots)
result = job.result()
counts = result.get_counts(circ)

postselectedtotcounts = 0
postselectedcounts = dict()
for key in counts.keys():
    keysp = key.split(' ')
    if key[0]=='1':
        postselectedtotcounts = postselectedtotcounts + counts[key]
        if key not in postselectedcounts:
            postselectedcounts[keysp[-1]] = 0
        postselectedcounts[keysp[-1]] = postselectedcounts[keysp[-1]] + counts[key]

measuredpostselectionprob = 100*postselectedtotcounts/shots 

maxkey = 0
for key in postselectedcounts.keys():
    k = bintoint(key)
    if k > maxkey:
        maxkey = k
print('Measured Postselection Probability: ', measuredpostselectionprob, '%')
print('Statevector Simulation Postselection Probability: ', postselectionprob, '%')
measurementcounts = np.zeros(shape=(maxkey+1,1))

for key in postselectedcounts.keys():
    idx = bintoint(key)
    measurementcounts[idx] = postselectedcounts[key]
    
probabilityofmeas = measurementcounts/postselectedtotcounts
normalizedmeasuredstate = np.sqrt(probabilityofmeas)
print('Measurements out of', shots, 'shots:')
print(measurementcounts)
print('Measured Probability:')
print(probabilityofmeas)
print('Normalized Measured State:')
print(normalizedmeasuredstate)
print('Simulated Statevector:')
print(finalstatevector)
print('Actual Answer:')
print(actualans)
print('Proportionality constant between actual answer and simulated statevector:')
print(np.mean(np.abs(actualans)/np.abs(finalstatevector)))
print('(Statevector)*(Proportionality Constant):')
print(finalstatevector*np.mean(np.abs(actualans)/np.abs(finalstatevector)))
print('Proportionality constant between actual answer and normalized measured state:')
print(np.mean(np.abs(actualans)/np.abs(normalizedmeasuredstate)))
print('(Normalized Measured State)*(Proportionality Constant):')
print(normalizedmeasuredstate*np.mean(np.abs(actualans)/np.abs(normalizedmeasuredstate)))       
