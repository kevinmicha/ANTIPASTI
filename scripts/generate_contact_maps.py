from Bio.PDB import PDBParser
import numpy as np
import os
import sys

def compute_distance_matrix(pdb_file, ag_agnostic=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    heavy, light, antigen = get_heavy_light(pdb_file)
    # Alpha-C coordinates (heavy)
    alpha_carbon_coordinates = {}
    for model in structure:
        for chain in model:
            if chain.id == heavy:
                for residue in chain:
                    if 'CA' in residue and residue.id[0] == ' ' and residue.id[1] in range(1, 114): # EDIT: the second condition is to consider the variable region only (Chothia numbering). Remove if you want everything
                        alpha_carbon_coordinates[(chain.id, residue.id)] = residue['CA'].get_coord()
    
    # Alpha-C coordinates (light)
    for model in structure:
        for chain in model:
            if chain.id == light and light is not None:
                for residue in chain:
                    if 'CA' in residue and residue.id[0] == ' ' and residue.id[1] in range(1, 108): # EDIT: the second condition is to consider the variable region only (Chothia numbering). Remove if you want everything
                        alpha_carbon_coordinates[(chain.id, residue.id)] = residue['CA'].get_coord()

    # Alpha-C coordinates (antigen)
    for model in structure:
        for chain in model:
            if chain.id == antigen and antigen is not None and ag_agnostic is False:
                for residue in chain:
                    if 'CA' in residue and residue.id[0] == ' ':
                        alpha_carbon_coordinates[(chain.id, residue.id)] = residue['CA'].get_coord()

    # Compute distances and create matrix
    num_residues = len(alpha_carbon_coordinates)
    distance_matrix = np.zeros((num_residues, num_residues))

    for i, (_, coord1) in enumerate(alpha_carbon_coordinates.items()):
        for j, (_, coord2) in enumerate(alpha_carbon_coordinates.items()):
            distance_matrix[i, j] = np.linalg.norm(coord1 - coord2)

    return distance_matrix

def get_heavy_light(path):
    h_chain_key = 'HCHAIN'
    l_chain_key = 'LCHAIN'
    antigen_key = 'AGCHAIN'

    with open(path, 'r') as f: 
        line = f.readlines()[3]
        if line.find(h_chain_key) != -1:
            h_pos = line.find(h_chain_key) + len(h_chain_key) + 1
            h_chain = line[h_pos:h_pos+1]
        else:
            h_chain = None
        if line.find(l_chain_key) != -1:
            l_pos = line.find(l_chain_key) + len(l_chain_key) + 1
            l_chain = line[l_pos:l_pos+1]
        else:
            l_chain = None
        if line.find(antigen_key) != -1:
            ag_pos = line.find(antigen_key) + len(antigen_key) + 1
            ag_chain = line[ag_pos:ag_pos+1]
        else:
            ag_chain = None        

    return h_chain, l_chain, ag_chain

# Saving
def save_distance_matrix(distance_matrix, output_file):
    np.save(output_file, distance_matrix)

args = sys.argv[1:]

distance_matrix = compute_distance_matrix(args[0])
if args[2] == 'all':
    contact_map = distance_matrix
else:
    contact_map = np.where(distance_matrix<=float(args[2]), 1, 0)

save_distance_matrix(contact_map, args[1])