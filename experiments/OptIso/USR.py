from rdkit import Chem
import numpy as np
from scipy.spatial import distance
from scipy.stats import skew
import os
from hsr.pre_processing import *
import rdkit
from rdkit.Chem.rdMolDescriptors import GetUSRScore, GetUSR
from experiments.usr import *

cwd = os.getcwd()

np.set_printoptions(precision=4, suppress=True)

print(f'\nContinuity (USR): \n')
sorted_files = sorted(os.listdir(f'{cwd}/experiments/OptIso/molecules'))
for file in sorted_files:
    if file.endswith('.sdf'):
        molecules = load_molecules_from_sdf(f'{cwd}/experiments/OptIso/molecules/{file}', removeHs=False, sanitize=False)
    
    print(f'\n{file[:-4]}:\n')
    print(f'Reference points of the molecules:\n')
    n_molecules = len(molecules)
    for i in range(n_molecules):
        for j in range(i+1, n_molecules):
            similarity = compute_similarity(molecules[i], molecules[j],print_atom_number=True)
            print(f"\n(in-house) USR: {similarity:.4f}")

    # Compute USR similarity (rdkit function) between all pairs of molecules
    usrs = [GetUSR(mol) for mol in molecules]
    n_molecules = len(usrs)
    for i in range(n_molecules):
        for j in range(i+1, n_molecules):
            similarity = GetUSRScore(usrs[i], usrs[j])
            print(f"(rdkit) USR: {similarity:.4f}\n")