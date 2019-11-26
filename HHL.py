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

def hermtocontU(mat):
    # takes uncontrolled matrix
    # must be hermitian currently
    # - add code to make non-hermitian
    # matrix hermitian
    if np.array_equal(mat, mat.conj().T):
        hermop = mat
    T = 2 # how to pick T? 
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

# need to understand role of registers M and L
# and implement them

# fidelity of answer goes up with r
# probability of ancilla = 1 for post selection goes down with r
r=6
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
zeros='0'*binlen

postselectionprob = 0
postselectedvector = np.zeros(shape=(int(len(statevec)/2),1),dtype=np.complex_)
postselectedbinaryidx = list()

print('Full Statevector:')
for i in range(len(statevec)):
    binary = str(bin(i))[2:]
    if len(binary)<binlen:
        binary = zeros[:-len(binary)]+binary
    print(binary, statevec[i][0])
    if binary[-1]=='1':
        postselectionprob=postselectionprob+\
                np.sqrt(np.real(statevec.copy()[i][0])**2+\
                np.imag(statevec.copy()[i][0])**2)
        postselectedvector[int((i-1)/2)] = statevec[i][0]
        postselectedbinaryidx.append(binary)
postselectedvector= postselectedvector/postselectionprob
print('\n')

print('Postselected Statevector (Postselection prob - {:.2f}%):'.format(postselectionprob*100))
qbtoxstate = list()
qbtoxbinidx = list()
for i in range(len(postselectedvector)):
    print(postselectedbinaryidx[i][0:-1],postselectedvector[i][0])
    if postselectedbinaryidx[i][len(qbtox):len(qbtox)+len(qclock)]=='0'*len(qclock):
        qbtoxbinidx.append(postselectedbinaryidx[i][:len(qbtox)])
        qbtoxstate.append(postselectedvector[i][0])
print('\n')

print('Solution Statevector:')
for i in range(len(qbtoxstate)):
    print(qbtoxbinidx[i],qbtoxstate[i]*3)
print('\n')

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
print('\n############################')
print('### Measurement analysis ###')
print('############################\n')

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
print('-----------------------------------------------------------')
print('probability of ancilla = 1 for post-selection from measurement: ', 100*totcounts/shots, '%')
#print('-----------------------------------------------------------')
#print('percent probabilities of qubits = 1, conditioned on ancilla = 1:\n', countpercent)


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
