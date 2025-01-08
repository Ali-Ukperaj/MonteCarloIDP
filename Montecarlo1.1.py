import os
import copy
import scipy
import warnings
import numpy as np
import matplotlib.pyplot as plt

# scikit-learn gives an improper version warning
warnings.filterwarnings('ignore', category=UserWarning)
# Seq_Comp gives a deprecation warning for the way it is solved, however, it gives the right values
warnings.filterwarnings('ignore', category=DeprecationWarning)

## Import and define relevant data/constants
Amino_1 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M',
           'F','P','S','T','W','Y','V']
Amino_3 = ["Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His","Ile","Leu","Lys",
           "Met","Phe","Pro","Ser","Thr","Trp","Tyr","Val"]

#########################  User Inputs  ###########################
# Insert Ideal Input or "None" if None
Ideal_Seq = list('MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGG'
                 'YGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS')  #  FUS
# Ideal_Seq = list('EIEGCYDEVGVRGSKGFQPTSSNCVMLGKIRKQDTGSSKSTSAKWQRLSRIDSSPESASTVKPVNAPKNEHSPTDSTG'
#                  'FKFGEPVSPTLATFLGESAFSAMTTQAFLLLSNPCPINKVSS')  #  Nup-120

## 3 length variables in this code. Inp_SeqLength is the length of the input sequence that a user enters and wants their
## sequence to be compositionally similar to. DesLength is the optional length they've input for how long they want their
## final sequence to be. Length is the local variable used in parameter calculation functions and is the length of
## whatever sequence is passed into the function. In this way, this length will be relative to whatever sequence is
## put into the function
Inp_SeqLength = len(Ideal_Seq)

# Insert Input Length
DesLength = 'None'

if DesLength == 'None':
    DesLength = Inp_SeqLength

# Define Comp_Goal and composition of sequence depending on input Ideal_Seq
if Ideal_Seq == 'None':
    Composition = 'None'
    Comp_Goal = 'None'
else:
    # Create a dictionary composition by finding the fraction of each amino acid in the reference sequence (n_amino/n_total)
    Composition = {amino: (len(np.where(np.array(Ideal_Seq) == amino)[0].tolist()))/(Inp_SeqLength) for amino in Amino_1}
    Comp_Goal = 0

# Desired values of parameters

# SCD_Goal, SHD_Goal, SAD_Goal, delG_Goal = -0.3, 5.0710, 0.172, -10  # Nup-120 Goals
SCD_Goal, SHD_Goal, SAD_Goal, delG_Goal = 0.0393, 6.5, 0.0649, -15  # FUS-WT LC Goals
Goal_Array = [SCD_Goal, SHD_Goal, SAD_Goal, delG_Goal, Comp_Goal]

# Desired ratio of weights of importance per parameter
InputSCD, InputSHD, InputSAD, InputdelG, InputComp = 1, 1, 1, 1, 1
InputWeights = [InputSCD, InputSHD, InputSAD, InputdelG, InputComp]
InputWeightDictionary = {'SCD': InputSCD, 'SHD': InputSHD, 'SAD': InputSAD, 'delG': InputdelG, 'Comp': InputComp}

# Desired ratio of mutations to swaps to shuffles
mutRatio = 1
swpRatio = 1
shfRatio = 1

# maximum amount of times we will cycle through the loop
desired_cycles = 1000
###################################################################

# Residue information
hydropathy = np.loadtxt('AA_properties/Urry.dat', dtype=object)
hydropathy[:, 1] = hydropathy[:, 1].astype(float)
S = np.loadtxt('AA_properties/Saltout.dat', dtype=object)
S[:, 1] = S[:, 1].astype(float)
c_salt = 0.1
hydropathy[:, 1] = hydropathy[:, 1] + (S[:, 1] * (c_salt - 0.1))
hydropathy = dict(hydropathy)
charge = np.loadtxt('AA_properties/charges.dat', dtype=object)
charge[:, 1] = charge[:, 1].astype(float)
charge = dict(charge)
mass = np.loadtxt('AA_properties/masses.dat', dtype=object)
mass[:, 1] = mass[:, 1].astype(float)
mass = dict(mass)
bAromatic = np.loadtxt('AA_properties/aromaticity.dat', dtype=object)
bAromatic[:, 1] = bAromatic[:, 1].astype(int)
bAromatic = dict(bAromatic)


####################################### Delta G Calculation #######################################
import pandas as pd
from predictor import *                                   # predictor.py must be in the same directory as this script
import joblib
residues = pd.read_csv('residues.csv').set_index('one')   # residue.csv must be in the same directory as this script
nu_file = 'svr_model_nu.joblib'                           # svr_model_nu.joblib must be in the same directory as this script
features = ['mean_lambda', 'faro', 'shd', 'ncpr', 'fcr', 'scd', 'ah_ij','nu_svr']
models = {}
models['dG'] = joblib.load(f'model_dG.joblib')          # model_dG.joblib must be in the same directory as this script
SEQUENCE = Ideal_Seq.copy()
SEQUENCE = str(SEQUENCE)
SEQUENCE = SEQUENCE.replace('[','').replace(']','').replace(',','').replace('\'','').replace(' ','')
CHARGE_TERMINI = True # @param {type:'boolean'}
seq = SEQUENCE
if " " in seq:
    seq = ''.join(seq.split())
    print('Blank character(s) found in the provided sequence. Sequence has been corrected, but check for integrity.')

def delta_G(seq,features,residues,nu_file):
    X = X_from_seq(seq,features,residues=residues,charge_termini=CHARGE_TERMINI,nu_file=nu_file)
    ys = models['dG'].predict(X)
    ys_m = np.mean(ys)
    return ys_m #kT
######################################### End Delta G Calculation ##################################

def SCD_calc(Seq):
    Length = len(Seq)
    ## Calculate charge for each residue
    q_Sequence = np.array([charge[Seq[i]] for i in range(Length)])

    ## From sequences, calculate net charge, NCPR, and fraction of +/- residues
    q_Net = q_Sequence.sum()
    NCPR = q_Net / Length
    fplus = (q_Sequence > 0).mean()
    fminus = (q_Sequence < 0).mean()

    ## From sequences, calculate the sequence charge decoration (SCD)
    SCD = 0
    for i in range(1, Length):
        SCD += ((q_Sequence[i:] * q_Sequence[:-i]) * np.sqrt(i)).sum()
    SCD /= Length
    return SCD

def SHD_calc(Seq):
    Length = len(Seq)
    ## Calculate hydropathy for each residue
    h_Sequence = np.array([hydropathy[Seq[i]] for i in range(Length)])
    ## From sequences, calculate average hydropathy
    h_Avg = h_Sequence.mean()
    std_h = np.std(h_Sequence)

    ## From sequences, calculate the sequence hydrophobicity decoration (SHD)
    SHD = 0
    for i in range(1, Length):
        SHD += ((h_Sequence[i:] + h_Sequence[:-i]) / i).sum()
    SHD /= Length
    return SHD

def SAD_calc(Seq):
    Length = len(Seq)
    ## Calculate aromaticity for each residue
    a_Sequence = np.array([bAromatic[Seq[i]] for i in range(Length)])
    ## From sequences, calculate average aromaticity
    a_Avg = a_Sequence.mean()

    ## From sequences, calculate the sequence aromatic decoration (SAD)
    SAD = 0
    for i in range(1, Length):
        SAD += ((a_Sequence[i:] * a_Sequence[:-i]) / i).sum()
    SAD /= Length
    return SAD

# Function that takes a sequence and returns ALL desired parameters
def param_calc(Seq):
    Length = len(Seq)
    # Define SCD, SHD, SAD, delta(G), and composition RMSD of the sequence
    if InputSCD == 'None':
        SCD = 'None'
    else:
        SCD = SCD_calc(Seq)

    if InputSHD == 'None':
        SHD = 'None'
    else:
        SHD = SHD_calc(Seq)

    if InputSAD == 'None':
        SAD = 'None'
    else:
        SAD = SAD_calc(Seq)

    if InputdelG == 'None':
        delG = 'None'
    else:
        delG = delta_G(Seq,features,residues,nu_file)

    # Define Composition RMSD
    if InputComp == 'None':
        Comp_RMSD = 'None'
    else:
        # Calculate fraction of each amino in the sequence tested
        Seq_Comp = {amino: ((len(np.where(np.array(Seq) == amino)[0].tolist())) / (Length)) for amino in Amino_1}
        Comp_RMSD = 0
        for amino_name, frac_amino in Composition.items():
            # Uses common dictionary keys (Amino 1-Letter Abbrev.) to compare the fraction of aminos in ideal vs input sequence
            Comp_RMSD += (Seq_Comp[amino_name] - frac_amino)**2

    params = {'SCD': SCD,
              'SHD': SHD,
              'SAD': SAD,
              'delG': delG,
              'Comp': Comp_RMSD}
    return params

# Define energy function of a sequence, using other param_calc func. to calculate their parameters
def energy_func(Seq):
    global Goals
    # Params is array with following format: [SCD, SHD, SAD, delG]
    params = param_calc(Seq)
    energy = 0
    for key, (goal, weight) in Goals.items():
        if params[key] != 'None':
            energy += weight*(abs(params[key] - goal))
    return energy

# Calculate Metropolis criterion by taking two sequences and using the energy_func function to calculate their energy
def Metropolis(Seq0, SeqMut):
    k_B = 1
    T = 10**(-2)
    energy0 = energy_func(Seq0)
    energyMut = energy_func(SeqMut)
    deltaE = energyMut - energy0
    Metrop = scipy.special.expit(-deltaE/(k_B*T))
    return Metrop

####################  Type of alterations  ####################
def seq_mut(Seq):
    # determines place in chain where mutation will occur
    idx = np.random.randint(0, len(Seq))  # integer that will be the index for mutated residue in sequence
    Seq_New = copy.deepcopy(Seq[:])  # Create new sequence to not affect original input sequence
    # create new Amino list that doesn't include residue currently there
    Amino_1_New = [res for res in Amino_1 if res != Seq[idx]]
    Seq_New[idx] = np.random.choice(Amino_1_New)  # assign random residue to new sequence in determined spot
    return Seq_New

def seq_swap(Seq):
    idx1 = np.random.randint(0, len(Seq))  # integer that will be the index for mutated residue in sequence
    Amino_idx1 = Seq[idx1]
    idx2 = np.random.randint(0, len(Seq))  # integer that will be the index for mutated residue in sequence
    Amino_idx2 = Seq[idx2]
    Seq_New = copy.deepcopy(Seq[:])  # Create new sequence to not affect original input sequence
    # create new Amino list that doesn't include residue currently there
    Seq_New[idx1] = Amino_idx2
    Seq_New[idx2] = Amino_idx1
    return Seq_New

# Function that will shuffle a variable length of the sequence ranging from (3, N)
def seq_shuf(Seq):
    # Input sequence MUST be a numpy array containing a list. This is because the sequences will be indexed with another list and
    # this functionality is only possible with a numpy array
    copyShuf = copy.deepcopy(Seq[:])
    lenseq = len(Seq)
    shuf_size = np.random.randint(3, lenseq)
    shuf_start = np.random.randint(0, lenseq)
    shuf_indices = np.linspace(shuf_start, shuf_start+shuf_size-1, shuf_size)
    indices = []
    # conditional that makes indices periodic by catching out of bounds indices
    for ind in shuf_indices:
        if ind > (lenseq-1):
            indices.append(int(ind)-int(lenseq))
        else:
            indices.append(int(ind))
    Seq_New = copy.deepcopy(copyShuf[:])
    Seq_New[indices] = np.random.permutation(copyShuf[indices])
    return Seq_New
###############################################################


#################### Choosing Alteration ######################
def alteration(mutAmt, swpAmt, shfAmt, Seq):
    global Amino_1
    Amino_1 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
               'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    lowbound = 0
    upbound = mutAmt + swpAmt + shfAmt
    coin_flip = np.random.randint(lowbound, upbound)
    if (coin_flip >= lowbound) & (coin_flip < (lowbound + mutAmt)):
        New_Seq = seq_mut(Seq)
        seqIdentifier = 0  # variable used to identify what mutation occurred.
    elif (coin_flip >= (lowbound + mutAmt)) & (coin_flip < (lowbound + mutAmt + swpAmt)):
        New_Seq = seq_swap(Seq)
        seqIdentifier = 1
    elif (coin_flip >= (lowbound + mutAmt + swpAmt)) & (coin_flip < (lowbound + mutAmt + swpAmt + shfAmt)):
        New_Seq = seq_shuf(Seq)
        seqIdentifier = 2
    return New_Seq, seqIdentifier
###############################################################

##################### Param Weights Calc #######################
def SHD_Weight(Seq, Length):
    hydropathy = np.loadtxt('AA_properties/Urry.dat', dtype=object)
    hydropathy[:, 1] = hydropathy[:, 1].astype(float)
    h_Sequence_orig = np.array([hydropathy[np.where(Seq[m] == hydropathy)[0], 1] for m in range(Length)])
    std_hyd = np.std(h_Sequence_orig)
    est_stdSHD = (0.65 * (Length ** (-0.45922))) * std_hyd
    WtSHD = 1/est_stdSHD
    return WtSHD
################################################################


######################  Include recommended weights based on separate calculations  ######################
# Calculated Weights
SCD_Weight, SHD_Weight, SAD_Weight, delG_Weight, Comp_Weight = 0.1, SHD_Weight(Ideal_Seq, Inp_SeqLength)/4, 50, 100/3, 10000

CalcWeights = [SCD_Weight, SHD_Weight, SAD_Weight, delG_Weight, Comp_Weight]
##########################################################################################################

# Initialize final weights
Weights = [0, 0, 0, 0, 0]
# Use input weight ratios to balance the calculated and input weights & define Goals + Intervals
for count, inp in enumerate(InputWeights):
    if inp == 'None':
        Weights[count] = 0
    else:
        Weights[count] = inp*CalcWeights[count]

# Set intervals
Intervals = [0, 0, 0, 0, 0]
for count, goal in enumerate(Goal_Array):
    if goal == 0:
        Intervals[count] = 0.0005
    elif goal == "None":
        Intervals[count] = "None"
    else:
        Intervals[count] = abs(goal*0.0005)

Intervals_Dict = {
    'SCD': Intervals[0],
    'SHD': Intervals[1],
    'SAD': Intervals[2],
    'delG': Intervals[3],
    'Comp': Intervals[4]}
# Save keys where Interval != None
saved_Keys = []
for key, inte in InputWeightDictionary.items():
    if inte != "None":
        saved_Keys.append(key)

# Place parameter goals and their associated weights in a dictionary format
Goals = {
    'SCD': [SCD_Goal, Weights[0]],
    'SHD': [SHD_Goal, Weights[1]],
    'SAD': [SAD_Goal, Weights[2]],
    'delG': [delG_Goal, Weights[3]],
    'Comp': [Comp_Goal, Weights[4]]}

# Determine starting sequence by minimizing the energy function out of 100 randomly generated sequences
# Initialize array for saving pairs of sequences and their respective "energies"
# ** If user wants no mutations, skip this step
if mutRatio == 0:
    Seq = Ideal_Seq
else:
    MinEnergySeq = np.empty(shape=[100, 2], dtype=object)
    for i in range(100):
        SeqRand = np.random.choice(Amino_1, DesLength)
        energyRand = energy_func(SeqRand)
        MinEnergySeq[i] = ["".join(SeqRand), energyRand]
    # Sequence with the minimum energy
    Seq = np.array(list(MinEnergySeq[np.argmin(MinEnergySeq[:, 1], axis=0), 0]))

# Set starting sequence's parameters and initialize the iterations variable and output array for tracking param movement
params = param_calc(Seq)
Seq_Comp = {amino: ((len(np.where(np.array(Seq) == amino)[0].tolist())) / (DesLength)) for amino in Amino_1}
SCD, SHD, SAD, delG, Comp = params['SCD'], params['SHD'], params['SAD'], params['delG'], params['Comp']

if (desired_cycles >= 100) & (desired_cycles % 100 == 0):
    snapshot_freq = desired_cycles//100  # how often we're tracking the progress of our parameters and sequence
elif (desired_cycles >= 100) & (desired_cycles % 100 != 0):
    snapshot_freq = 10
else:
    snapshot_freq = 1

# Will be used to store sequence and all other desired parameters
Movement = np.empty(shape=[(desired_cycles//snapshot_freq)+1, 6], dtype=object)
iterations, frame = 0, 0

# Set counters for number of specific types of alterations and their respective accepted changes.
mutCount, mutAccepted, swpCount, swpAccepted, shfCount, shfAccepted = 0, 0, 0, 0, 0, 0

def check_conditions():
    # Determine form of conditionals
    if len(saved_Keys) == 1:
        idx1 = saved_Keys[0]
        cond = (abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]
    elif len(saved_Keys) == 2:
        idx1, idx2 = saved_Keys[0], saved_Keys[1]
        cond = (((abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]) or
                ((abs(params[idx2] - Goals[idx2][0])) > Intervals_Dict[idx2]))
    elif len(saved_Keys) == 3:
        idx1, idx2, idx3 = saved_Keys[0], saved_Keys[1], saved_Keys[2]
        cond = ((((abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]) or
                ((abs(params[idx2] - Goals[idx2][0])) > Intervals_Dict[idx2])) or
                ((abs(params[idx3] - Goals[idx3][0])) > Intervals_Dict[idx3]))
    elif len(saved_Keys) == 4:
        idx1, idx2, idx3, idx4 = saved_Keys[0], saved_Keys[1], saved_Keys[2], saved_Keys[3]
        cond = (((((abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]) or
                 ((abs(params[idx2] - Goals[idx2][0])) > Intervals_Dict[idx2])) or
                ((abs(params[idx3] - Goals[idx3][0])) > Intervals_Dict[idx3])) or
                ((abs(params[idx4] - Goals[idx4][0])) > Intervals_Dict[idx4]))
    elif len(saved_Keys) == 5:
        idx1, idx2, idx3, idx4, idx5 = saved_Keys[0], saved_Keys[1], saved_Keys[2], saved_Keys[3], saved_Keys[4]
        cond = ((((((abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]) or
                  ((abs(params[idx2] - Goals[idx2][0])) > Intervals_Dict[idx2])) or
                 ((abs(params[idx3] - Goals[idx3][0])) > Intervals_Dict[idx3])) or
                ((abs(params[idx4] - Goals[idx4][0])) > Intervals_Dict[idx4])) or
                ((abs(params[idx5] - Goals[idx5][0])) > Intervals_Dict[idx5]))
    return cond

# Set combination of conditionals based on what parameters are fed into the script
print(f"Original Sequence: {''.join(Seq)}\n")
# Loop that runs until ALL parameters are within their proper interval OR iterations reaches the desired # of cycles
while (check_conditions() & (iterations <= desired_cycles)):

    # Update what iteration you are on every 1/10th of the maximum amount of cycles
    if iterations % (desired_cycles/10) == 0:
        print(f"Current iteration: {iterations}\n")

    # Take "snapshot" of data every 1000th step
    if iterations % snapshot_freq == 0:
        Movement[frame] = ["".join(Seq), SCD, SHD, SAD, delG, Comp]
        frame += 1  # counter keeping track of what "frame" we're currently on

    # Set conditional that determines if we run a mutation or swap (mutation == 0; swap == 1)
    Seq_New, chngIdent = alteration(mutRatio, swpRatio, shfRatio, Seq)

    acceptedBool = False
    # Determine Metropolis criterion of this sequence change
    Metrop = Metropolis(Seq, Seq_New)
    if Metrop >= 1:
        # Accept the change by making the "Seq" variable identical to "Seq_New"
        Seq = copy.deepcopy(Seq_New)
        # Since alteration is accepted, we must recalculate new parameters to update variables in loops conditionals.
        params = param_calc(Seq)
        SCD, SHD, SAD, delG, Comp = params['SCD'], params['SHD'], params['SAD'], params['delG'], params['Comp']
        acceptedBool = True
    elif Metrop < 1:
        # generate another random number between 0 and 1 to determine if sequence mutation/swap is accepted
        coin_flip2 = np.random.uniform(0, 1)
        # If this number is less than or equal to the Metropolis (less than one), accept the sequence alteration
        if coin_flip2 <= Metrop:
            Seq = copy.deepcopy(Seq_New)
            # Since alteration is accepted, we must recalculate new parameters to update variables in loops conditionals.
            params = param_calc(Seq)
            SCD, SHD, SAD, delG, Comp = params['SCD'], params['SHD'], params['SAD'], params['delG'], params['Comp']
            acceptedBool = True
    if (chngIdent == 0):
        mutCount += 1
        if (acceptedBool):
            mutAccepted += 1
    elif (chngIdent == 1):
        swpCount += 1
        if (acceptedBool):
            swpAccepted += 1
    elif (chngIdent == 2):
        shfCount += 1
        if (acceptedBool):
            shfAccepted += 1

    # Counts iterations for total number of mutations in script
    iterations += 1

print(f"Final, optimized sequence: {''.join(Seq)}\n")
# Output array to save final sequence and resultant parameters: [Seq, SCD, SHD, SAD, delG, Composition RMSD]
output = ["".join(Seq), params['SCD'], params['SHD'], params['SAD'], params['delG'], params['Comp']]
np.savetxt('Coordinates', output, fmt='%s', delimiter=',')

# Save Information
file_save = 'Total_Movements'
new_save = copy.deepcopy(file_save)
count = 0
while os.path.exists(new_save):
    count += 1
    new_save = f"{file_save}{count}"
file_save = copy.deepcopy(new_save)
np.savetxt(file_save, Movement, fmt='%s', delimiter=',')


print(f"The number of iterations: {iterations}\n\n")
print(f"Output Array [Seq, SCD, SHD, SAD, delG, Composition RMSD]:\n {output[1:6]}")

for i in range(1, 6):
    if Movement[0, i] != "None":
        Movement[:, i] = Movement[:, i].astype(float)

total_Moves = mutAccepted + swpAccepted + shfAccepted

Steps = snapshot_freq*(np.arange(0, frame, 1))
print(f"Movements: {snapshot_freq*len(Movement[:frame, 1])}\n Steps: {snapshot_freq*len(Steps)}\n")
print(f"MutAtt: {mutCount}\n MutAcc: {mutAccepted}\n")
print(f"SwpAtt: {swpCount}\n SwpAcc: {swpAccepted}\n")
print(f"ShfAtt: {shfCount}\n ShfAcc: {shfAccepted}\n")
print(f"Moves Attempted: {iterations}\nMoves Accepted: {total_Moves}")

# Plot movement of Parameters
if Movement[0, 1] != "None":
    plt.figure(1)
    plt.plot(Steps, Movement[:frame, 1], color='k')
    plt.title(f"SCD")
    plt.xlabel('Iterations')
    plt.savefig('SCD_plots120.png')
    plt.show()

if Movement[0, 2] != "None":
    plt.figure(2)
    plt.plot(Steps, Movement[:frame, 2], color='k')
    plt.title(f"SHD")
    plt.xlabel('Iterations')
    plt.savefig('SHD_plots120.png')
    plt.show()

if Movement[0, 3] != "None":
    plt.figure(3)
    plt.plot(Steps, Movement[:frame, 3], color='k')
    plt.title(f"SAD")
    plt.xlabel('Iterations')
    plt.savefig('SAD_plot120.png')
    plt.show()

if Movement[0, 4] != "None":
    plt.figure(4)
    plt.plot(Steps, Movement[:frame, 4], color='k')
    plt.title(f"deltaG")
    plt.xlabel('Iterations')
    plt.savefig('deltaG_plot120.png')
    plt.show()

if Movement[0, 5] != "None":
    plt.figure(5)
    plt.plot(Steps, Movement[:frame, 5], color='k')
    plt.title(f"Composition RMSD")
    plt.xlabel('Iterations')
    plt.savefig('CompRMSD_plots120.png')
    plt.show()
