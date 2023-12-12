import numpy as np

from config import DATA_DIR
    
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
            print('loser')
            print(pdb_code)
            
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
    numerical_values: list
        If data is numerical (e.g., affinity values), it is necessary to include a list here. In this way, values associated to nanobodies can be removed.

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

