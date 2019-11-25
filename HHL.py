from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info.operators import Operator
from qiskit import Aer
import numpy as np
from math import pi
from scipy.linalg import expm

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

def hermtocontU(mat):
    # takes uncontrolled matrix
    # must be hermitian currently
    # - add code to make non-hermitian
    # matrix hermitian
    if np.array_equal(mat, mat.conj().T):
        hermop = mat
    T = 2 # how to pick T? larger T bitshifts result 
          # is final result (returned values)*T??
    expherm = expm(2j*pi*hermop/T)
    # add control
    M0 = np.asarray([[1,0],\
                     [0,0]])
    M1 = np.asarray([[0,0],\
                     [0,1]])
    I = np.eye(np.shape(hermop)[0])
    cexpherm = np.kron(M0,I)+np.kron(M1,expherm)

    #what if cexpherm dimensions are not a power of 2?
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
b = np.asarray([2,0]) 

'''
# From the paper, 'Quantum Circuit Design for Solving 
# Linear Systems of Equations'
A = np.asarray([[15, 9, 5, -3],\
                [9, 15, 3, -5],\
                [5, 3, 15, -9],\
                [-3, -5, -9, 15]])
b = np.asarray([1,1,1,1])
'''
cexpherm = hermtocontU(A)
clocksize = 2

# qbtox size wont work if b and cepherm dimensions are not a power of 2
# need to modify hermtocontU() and prepareb() to make sure they produce 
# stuff with dimension 2^n by 2^n. this may change how the result is to be
# interpreted, so hermtocontU() should also return flags indicating
# how to proceed in later parts of the algorithm

qclock = QuantumRegister(clocksize,'clk')
qbtox = QuantumRegister(np.log2(np.shape(b)[0]),'btox')
qanc = QuantumRegister(1,'anc')
cclock = ClassicalRegister(clocksize,'cclk')
cbtox = ClassicalRegister(np.log2(np.shape(b)[0]),'cbtox')
canc = ClassicalRegister(1,'canc')

circ = QuantumCircuit(qbtox,qclock,qanc,cbtox,cclock,canc)

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

# need to understand role of registers M and L
# and implement them

# fidelity of answer goes up with r
# probability of ancilla = 1 for post selection goes down with r
r=5
for i in range(len(qclock)):
    circ.cry((2**(len(qclock)-1-i)*pi)/2**(r),qclock[i],qanc[0])
circ.barrier()

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

#########################################################
### get statevector for qbtox conditioned on qanc = 1 ###
#########################################################

statevec = getstatevector(circ)

# make measurement operators ??

#####################################
### measure, analyze measurements ###
#####################################
circ.measure(qanc,canc)
circ.measure(qbtox,cbtox)
circ.measure(qclock,cclock)

print(circ)

shots = 100000
bkend = Aer.get_backend('qasm_simulator')
job = execute(circ, bkend, shots=shots)
result = job.result()
counts = result.get_counts(circ)

'''
#QPE analysis
values = dict()
vectors = list()
for key in counts.keys():
    if key.split(' ')[1] not in values:
        values[key.split(' ')[1]]=np.zeros(shape=(2**len(qbtox),1))
        print(key, counts[key])
print('\n')

for key in counts.keys():
    val = key.split(' ')[1]
    count = counts[key]
    index = bintoint(key.split(' ')[2])
    values[val][index] = count 

for key in values.keys():
    values[key] = values[key]/np.sum(values[key])
    print(key, binfractodec(key), \
            np.around(np.exp(2j*pi*binfractodec(key)),decimals=4),\
            values[key])
'''


# Get all qubit probabilities, for comparison with Quirk
print('\n\n')
countpercent = np.zeros(shape=(len(qanc)+len(qbtox)+len(qclock),1))
totcounts = 0
for key in counts.keys():
    keysp = key.split(' ')
    if keysp[0] == '1':
        totcounts = totcounts + counts[key]
        countpercent[0] = countpercent[0] + counts[key]
        for i in range(len(keysp[1])):
            if keysp[1][i]== '1':
                countpercent[1+i] = countpercent[1+i] + counts[key]
        for i in range(len(keysp[2])):
            if keysp[2][i] == '1':
                countpercent[1+len(keysp[1])+i] = countpercent[1+len(keysp[1])+i] + counts[key]

countpercent=100*countpercent/totcounts
print('probability of ancilla = 1 for post-selection: ', 100*totcounts/shots, '%')
print('percent probabilities of qubits = 1, conditioned on ancilla = 1:\n', countpercent)
print('\n\n')

# get probabilities
totcounts = 0
HHLans = np.zeros(shape=(2**len(qbtox),1))
for key in counts.keys():
    keysp = key.split(' ')
    if keysp[0] == '1': 
        totcounts = totcounts + counts[key]
        for i in range(len(qbtox)):
            HHLans[bintoint(keysp[2])] = counts[key]
HHLans = HHLans/totcounts

actualans=np.matmul(np.linalg.inv(A),np.asarray(b).reshape(len(b),1))
print('State probabilities from HHL:')
for i in range(np.shape(HHLans)[0]):
    print('{}|{}>'.format(HHLans[i][0],i), end=' ')
    if i < np.shape(HHLans)[0]-1:
        print('+', end=' ')

print('\n-----------------------------------------------------------')
print('Square root of these probabilities is proportional to (the magnitudes of) the exact solution, x:')
print('|x,HHL> =', np.sqrt(HHLans.T)[0])
print('-----------------------------------------------------------')
print('The actual solution is:')
print('x =', actualans.T[0], 'and the absolute value of x\'s elements should equal C|x,HHL>')
print('-----------------------------------------------------------')
print('|x|/|x,HHL> element-wise is')
print(np.abs(actualans)/np.sqrt(HHLans))
print('-----------------------------------------------------------')
print('We can take the average of these elements as C')
print('C =', np.mean(np.abs(actualans)/np.sqrt(HHLans)))
print('-----------------------------------------------------------')
print('Now we can show that |x,HHL> is proportional to the magnitudes of elements of x')
print('C|x,HHL> =', np.sqrt(HHLans.T)[0]*np.mean(np.abs(actualans)/np.sqrt(HHLans)), 'is approximately equal to absolute value of x\'s elements,', np.abs(actualans.T[0]))
print('-----------------------------------------------------------')
