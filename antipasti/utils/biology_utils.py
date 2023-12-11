import numpy as np
import requests
from bs4 import BeautifulSoup

from config import DATA_DIR

def get_types_of_residues(pdb_codes):
    r"""Returns lists of aromaticity, hydrophobicity, charge and polarity scores between 0 and 1.

    Parameters
    ----------
    pdb_codes: list
        The PDB codes of the antibodies.

    """
    aromaticity = []
    hydrophobicity = []
    charge = []
    polarity = []
    aromatic_residues = ['W', 'Y', 'F', 'H']
    aliphatic_hydrophobic_residues = ['A', 'V', 'L', 'I', 'P', 'M', 'C']
    charged_residues = ['D', 'E', 'K', 'R']
    polar_residues = ['N', 'Q', 'S', 'T']

    for pdb_code in pdb_codes:
        print(pdb_code)
        paratope = extract_paratope_epitope(pdb_code, 'Paratope')
        if paratope == '' or paratope[1] == '':
            aromaticity.append('unknown')
            hydrophobicity.append('unknown')
            charge.append('unknown')
            polarity.append('unknown')
        else:
            epitope = extract_paratope_epitope(pdb_code, 'Epitope')
            paratope_list = paratope[1].split()
            aromaticity.append(float(len([residue for residue in paratope_list if residue in aromatic_residues])/len(paratope_list)))
            hydrophobicity.append(float(len([residue for residue in paratope_list if residue in aliphatic_hydrophobic_residues])/len(paratope_list)))
            charge.append(float(len([residue for residue in paratope_list if residue in charged_residues])/len(paratope_list)))
            polarity.append(float(len([residue for residue in paratope_list if residue in polar_residues])/len(paratope_list)))

    return aromaticity, hydrophobicity, charge, polarity

def extract_paratope_epitope(pdb_code, region='Paratope'):
    r"""Retrieves the paratope and epitope members of an antibody from the IMGT database.

    Parameters
    ----------
    pdb_code: str
        The antibody PDB code.

    """
    url = f'https://www.imgt.org/3Dstructure-DB/cgi/details.cgi?pdbcode={pdb_code}&Part=Epitope'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    table_rows = soup.find_all('tr')
    residue_row = None
    for row in table_rows:
        if row.find('td', class_='titre_title') and region.lower() in row.text.lower():
            type_row = row.find_next_sibling('tr')
            residue_row = type_row.find_next_sibling('tr')
            res_chain_row = residue_row.find_next_sibling('tr')
            break

    if residue_row:
        type_text = type_row.find('td', class_='data_r').text.strip()
        residues_text = residue_row.find('td', class_='data_r').text.strip().replace('IMGT Residue@Position cards', '')
        res_chain_text = res_chain_row.find('td', class_='data_r').text.strip()
        return type_text, residues_text, res_chain_text
    else:
        return ''
    
def extract_mean_region_lengths(pdb_codes, data_path=DATA_DIR):
    r"""Retrieves the FR and CDR lengths of an antibody.

    Parameters
    ----------
    pdb_code: str
        The antibody PDB code.
    data_path: str
        Path to the data folder.

    """
    region_lengths = np.zeros((14))

    for pdb_code in pdb_codes:
        res_l = list(np.load(data_path+f'lists_of_residues/{pdb_code}.npy'))
        h = res_l[1][0]
        l = res_l[-2][0]
        
        # Problems beginning CDR-H1
        if h+' 26 ' in res_l:
            cdrh1_b = res_l.index(h+' 26 ')
        elif h+' 27 ' in res_l:
            cdrh1_b = res_l.index(h+' 27 ')
        elif h+' 28 ' in res_l:
            cdrh1_b = res_l.index(h+' 28 ')
        elif h+' 29 ' in res_l:
            cdrh1_b = res_l.index(h+' 29 ')
        else:
            cdrh1_b = res_l.index(h+' 30 ')

        # Problems beginning CDR-H2
        if h+' 52 ' in res_l:
            cdrh2_b = res_l.index(h+' 52 ')
        else:
            cdrh2_b = res_l.index(h+' 53 ')

        # Beginning of FR1 (light chain)
        cfr1l_b = res_l.index(next((item for item in res_l if item.startswith(l)), None))
            
        # Problems beginning CDR-L2
        if l+' 50 ' in res_l:
            cdrl2_b = res_l.index(l+' 50 ')
        elif l+' 51 ' in res_l:
            cdrl2_b = res_l.index(l+' 51 ')
        elif pdb_code in ['4hkx', '5d70', '5d71']:
            cdrl2_b = 0
            cdrl2_e = 0
        else:
            cdrl2_b = res_l.index(l+' 52 ')
        
        # Problems end CDR-L2
        if l+' 57 ' in res_l:
            cdrl2_e = res_l.index(l+' 57 ')

        frh_parts = [len(res_l[1:cdrh1_b]), len(res_l[res_l.index(h+' 33 '):cdrh2_b]), len(res_l[res_l.index(h+' 57 '):res_l.index(h+' 95 ')]), len(res_l[res_l.index(h+'103 '):cfr1l_b])]
        cdrh_parts = [len(res_l[cdrh1_b:res_l.index(h+' 33 ')]), len(res_l[cdrh2_b:res_l.index(h+' 57 ')]), len(res_l[res_l.index(h+' 95 '):res_l.index(h+'103 ')])]
        if l != h:
            cdrl_parts = [len(res_l[res_l.index(l+' 24 '):res_l.index(l+ ' 35 ')]), len(res_l[cdrl2_b:cdrl2_e]), len(res_l[res_l.index(l+' 89 '):res_l.index(l+ ' 98 ')])]
            frl_parts = [len(res_l[cfr1l_b:res_l.index(l+' 24 ')]), len(res_l[res_l.index(l+ ' 35 '):cdrl2_b]), len(res_l[cdrl2_e:res_l.index(l+' 89 ')]), len(res_l[res_l.index(l+ ' 98 '):-1])]
        else:
            cdrl_parts = [0, 0, 0]
            frl_parts = [0, 0, 0, 0]

        for i in range(4):
            region_lengths[2*i] += frh_parts[i] / len(pdb_codes)
            region_lengths[2*i+7] += frl_parts[i] / len(pdb_codes)

            if i != 3:
                region_lengths[2*i+1] += cdrh_parts[i] / len(pdb_codes)
                region_lengths[2*i+8] += cdrl_parts[i] / len(pdb_codes)

    return region_lengths

def remove_nanobodies(pdb_codes, representations, embedding=None, labels=[], numerical_values=None):
    r"""Returns PDB codes and embeddings without the presence of nanobodies.

    Parameters
    ----------
    pdb_codes: list
        The PDB codes of the antibodies.
    representations: numpy.ndarray
        Normal mode correlation maps (or transformed maps) from which it can be inferred whether a given antibody is a nanobody.
    embedding: numpy.ndarray
        Low-dimensional version of ``representations``.
    labels: list
        Data point labels.

    """
    input_shape = representations.shape[-1]
    deleted_items = 0

    for i in range(len(pdb_codes)):
        if np.count_nonzero(representations[i-deleted_items].reshape(input_shape, input_shape)[-40:,-40:]) == 0:
            pdb_codes = np.delete(pdb_codes, i-deleted_items, axis=0)
            representations = np.delete(representations, i-deleted_items, axis=0)
            if embedding is not None:
                embedding = np.delete(embedding, i-deleted_items, axis=0)
            if len(labels):
                labels = np.delete(labels, i-deleted_items, axis=0)
            if numerical_values is not None:
                numerical_values = np.delete(numerical_values, i-deleted_items, axis=0)
            deleted_items += 1
    return pdb_codes, representations, embedding, labels, numerical_values

