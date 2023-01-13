import chemprop
from chemprop.features.featurization import map_reac_to_prod
from chemprop.rdkit import make_mol
import rdkit
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm
import pickle

import os
import json

import logging

def optimise_conformer(mol, logger):
    for _ in range(100):
        for func in (AllChem.MMFFOptimizeMoleculeConfs, AllChem.UFFOptimizeMoleculeConfs): 
            res = func(mol)

            if (min(res)[0] != -1):
                logger.debug(f"Successfully calculated energies at attempt {_} with function {func.__name__}")
                break
        
        if (min(res)[0] != -1):
                break

    if (min(res)[0] == -1):
        logger.debug("Could not calculate conformer energies")
        raise ValueError

    #@evan: Assume there are no 0s and -1s at the same time
    index = res.index(min(res))
    
    return index

def get_conformer(mol, logger):
    params = rdkit.Chem.rdDistGeom.ETKDGv3()
    params.randomSeed = 1
    params.useSmallRingTorsions = True

    num_conf = [200, 150, 100, 50, 20, 15, 10, 5, 1]
    indx = 0    
    while indx <= 8:
        try:
            conf_ids = AllChem.EmbedMultipleConfs(mol, num_conf[indx], params)
            logger.debug(f"Successfully embedded {num_conf[indx]} conformers")
            break
        except RuntimeError:
            indx += 1

            if indx == 9:
                return None
    
    try:
        index = optimise_conformer(mol, logger)
    except ValueError:
        try:
            logger.debug(f"Removing stereochemistry...")
            rdkit.Chem.RemoveStereochemistry(mol)
            index = optimise_conformer(mol, logger)            
        except ValueError:
            logger.debug(f"Failed to optimise conformer.")
            return None

    return mol.GetConformer(conf_ids[index])

def get_spatial_features(mol, logger):
    bond_distances = {}
    # bond_angles = {}

    fragments = rdkit.Chem.GetMolFrags(mol, asMols=True)

    co = 0
    for molecule in fragments:
        mol_conf = get_conformer(molecule, logger)
        if mol_conf == None:
            logger.debug("Could not get conformer!")
            return {} #Just void the whole SMILES string
            # continue

        for atom in molecule.GetAtoms():
            for atom2 in molecule.GetAtoms():
                dist = rdkit.Chem.rdMolTransforms.GetBondLength(mol_conf, atom.GetIdx(), atom2.GetIdx())
                bond_distances[(atom.GetIdx() + co, atom2.GetIdx() + co)] = dist
        
        co += molecule.GetNumAtoms()
    
    return bond_distances

def gen_dataset(filename, logger):
    reactions_upload_path = filename
    df = pd.read_csv(reactions_upload_path)

    mol_3d_features = {}

    if os.path.exists('all_3d_features.pkl'):
        with open('all_3d_features.pkl', 'rb') as f:
            mol_3d_features = pickle.load(f)

    for indx, mol in enumerate(tqdm(df['AAM'][:])):
        smiles_reac = mol.split(">")[0]
        smiles_prod = mol.split(">")[-1]
        
        if (smiles_reac not in mol_3d_features or mol_3d_features[smiles_reac] == {}):
            # mol_reac = make_mol(smiles_reac, True, False)
            mol_reac = make_mol(smiles_reac, True, True)
            logger.debug(f'{indx} reac: {smiles_reac}')
            mol_3d_features[smiles_reac] = get_spatial_features(mol_reac, logger)

        if (smiles_prod not in mol_3d_features or mol_3d_features[smiles_prod] == {}):
            # mol_prod = make_mol(smiles_prod, True, False)
            mol_prod = make_mol(smiles_prod, True, True)
            logger.debug(f'{indx} prod: {smiles_prod}')
            mol_3d_features[smiles_prod] = get_spatial_features(mol_prod, logger)

    with open('all_3d_features.pkl', 'wb') as f:
        pickle.dump(mol_3d_features, f)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('3d_logger.log')
    
    logger.addHandler(file_handler)

    datasets = ['datasets/rad6re.csv', 'datasets/phosphatase.csv']

    for dataset in datasets:
        try:
            gen_dataset(dataset, logger)
        except:
            logger.warning(f"FAIL FAIL FAIL: {dataset}")