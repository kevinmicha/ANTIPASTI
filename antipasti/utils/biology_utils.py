import numpy as np
import requests
from bs4 import BeautifulSoup

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
        paratope = extract_paratope_epitope(pdb_code, 'Paratope')
        if paratope == '' or paratope[1] == '':
            aromaticity.append('unknown')
            hydrophobicity.append('unknown')
            charge.append('unknown')
            polarity.append('unknown')
        else:
            epitope = extract_paratope_epitope(pdb_code, 'Epitope')
            paratope_list = paratope[1].split()
            aromaticity.append(len([residue for residue in paratope_list if residue in aromatic_residues])/len(paratope_list))
            hydrophobicity.append(len([residue for residue in paratope_list if residue in aliphatic_hydrophobic_residues])/len(paratope_list))
            charge.append(len([residue for residue in paratope_list if residue in charged_residues])/len(paratope_list))
            polarity.append(len([residue for residue in paratope_list if residue in polar_residues])/len(paratope_list))

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

def get_cdr_lengths(pdb_codes):
    r"""Returns lists with the lengths of the CDR-H1s, CDR-H2s, CDR-H3s, CDR-L1s, CDR-L2s and CDR-L3s. 

    Parameters
    ----------
    pdb_codes: list
        The PDB codes of the antibodies.

    """
    cdrl1_l = []
    cdrl2_l = []
    cdrl3_l = []
    cdrh1_l = []
    cdrh2_l = []
    cdrh3_l = []

    for pdb_code in pdb_codes:
        cdrl_parts, cdrh_parts = extract_cdr_lengths(pdb_code)
        
        cdrl1_l.append(cdrl_parts[0])
        cdrl2_l.append(cdrl_parts[1])
        cdrl3_l.append(cdrl_parts[2])
        cdrh1_l.append(cdrh_parts[0])
        cdrh2_l.append(cdrh_parts[1])
        cdrh3_l.append(cdrh_parts[2])

def extract_cdr_lengths(pdb_code):
    r"""Retrieves the CDR lengths of an antibody.

    Parameters
    ----------
    pdb_code: str
        The antibody PDB code.

    """
    res_l = list(np.load(f'data/paired_hl/lists_of_residues/{pdb_code}.npy'))
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
        
    cdrh_parts = [len(res_l[cdrh1_b:res_l.index(h+' 33 ')]), len(res_l[cdrh2_b:res_l.index(h+' 57 ')]), len(res_l[res_l.index(h+' 95 '):res_l.index(h+'103 ')])]
    if l != h:
        cdrl_parts = [len(res_l[res_l.index(l+' 24 '):res_l.index(l+ ' 35 ')]), len(res_l[cdrl2_b:cdrl2_e]), len(res_l[res_l.index(l+' 89 '):res_l.index(l+ ' 98 ')])]
    else:
        cdrl_parts = [0, 0, 0]

        
    return cdrl_parts, cdrh_parts
