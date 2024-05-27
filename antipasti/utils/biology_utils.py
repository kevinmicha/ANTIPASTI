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
        res_l = res_l[1:res_l.index('END-Ab')]
        res_l = [el[1:] for el in res_l] # Removing amino acid type
        h = res_l[0][0]
        l = res_l[-1][0]
        
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

def get_sequence(list_of_residues, max_res_list_h=None, max_res_list_l=None):
    r"""Returns an amino acid sequence from an ANTIPASTI list of residues. It contains gaps for the antibody.

    Parameters
    ----------
    list_of_residues: list
        Residues numbered according to the Chothia scheme with presence of 'START-Ab' and 'END-Ab' labels.
    max_res_list_h: list
        Heavy chain residues of all data.
    max_res_list_l: list
        Light chain residues of all data.

    """
    # First we force unique elements
    max_res_list_h = list(dict.fromkeys(max_res_list_h))
    max_res_list_l = list(dict.fromkeys(max_res_list_l))

    h_chain = list_of_residues[1][1]
    h = len([idx for idx in list_of_residues if idx[1] == h_chain])
    list_of_residues_h = list_of_residues[1:h+1]
    list_of_residues_l = list_of_residues[h+1:list_of_residues.index('END-Ab')]
    current_list_h = [x[2:].strip() for x in list_of_residues_h]
    current_list_l = [x[2:].strip() for x in list_of_residues_l]

    if max_res_list_h is None or max_res_list_l is None:
        sequence = [lor[0] for lor in list_of_residues[1:h+1]] # No antibody alignment
        sequence += [':'] # Separating chains
        if list_of_residues_l:
            sequence += [lor[0] for lor in list_of_residues[h+1:list_of_residues.index('END-Ab')]]
        sequence += [':'] # Separating chains
        sequence += [lor[0] for lor in list_of_residues[list_of_residues.index('END-Ab')+1:]]
    else:
        list_of_residues_iterator_h = iter(list_of_residues_h)
        list_of_residues_iterator_l = iter(list_of_residues_l)
        sequence = [next(list_of_residues_iterator_h, '-')[0] if max_res_list_h[i] in current_list_h else '-' for i in range(len(max_res_list_h))]
        sequence += [':'] # Separating chains
        if list_of_residues_l:
            sequence += [next(list_of_residues_iterator_l, '-')[0] if max_res_list_l[i] in current_list_l else '-' for i in range(len(max_res_list_l))]
        sequence += [':'] # Separating chains
        sequence += [lor[0] for lor in list_of_residues[list_of_residues.index('END-Ab')+1:]]

    return ''.join(map(str, sequence))

def antibody_sequence_identity(seq1, seq2):
    r"""Computes the percentage of sequence identity.

    Parameters
    ----------
    seq1: str
        First sequence.
    seq2: str
        Second sequence.
    
    """
    seq1 = seq1[:seq1.rfind(':')]
    seq2 = seq2[:seq2.rfind(':')]

    matching = sum(1 for ch1, ch2 in zip(seq1, seq2) if ch1 == ch2 and ch1 not in ('-', ':') and ch2 not in ('-', ':'))
    total = sum(1 for ch1, ch2 in zip(seq1, seq2) if ch1 not in ('-', ':') or ch2 not in ('-', ':'))

    return matching / total
    

def antigen_identity(seq1, seq2):
    r"""Tests whether two antibodies are bound to the same antigen.

    Parameters
    ----------
    seq1: str
        First sequence.
    seq2: str
        Second sequence.
    
    """
    return seq1.rsplit(':', 1)[1] == seq2.rsplit(':', 1)[1]

def check_train_test_identity(training_set_ids, test_set_ids, max_res_list_h=None, max_res_list_l=None, threshold=0.9, residues_path=DATA_DIR+'lists_of_residues/', verbose=False):
    r"""Tests the sequence identity of the training and test sets.

    Parameters
    ----------
    training_set_ids: list
        Contains the PDB identifiers of the training set elements.
    test_set_ids: list
        Contains the PDB identifiers of the test set elements.
    max_res_list_h: list
        Heavy chain residues of all data.
    max_res_list_l: list
        Light chain residues of all data.
    threshold: float
        Highest accepted sequence identity value.
    residues_path: str
        Path to the folder containing the list of residues per entry.

    """

    for test_element in test_set_ids:
        test_seq = get_sequence(list(np.load(residues_path+test_element+'.npy')), max_res_list_h=max_res_list_h, max_res_list_l=max_res_list_l)
        for training_element in training_set_ids:
            tr_seq = get_sequence(list(np.load(residues_path+training_element+'.npy')), max_res_list_h=max_res_list_h, max_res_list_l=max_res_list_l)
            identity = antibody_sequence_identity(tr_seq, test_seq)
            if identity > threshold:
                return False
            if antigen_identity(tr_seq, test_seq):
                return False

    print(f'All train/test pairs passed the similarity check (Identity <= {threshold:.2%})')
    return True

'''
def build_weights(pdb_codes, max_res_list_h=None, max_res_list_l=None, threshold=0.9, residues_path=DATA_DIR+'lists_of_residues/'):
    r"""Generates a vector that, for each sequence, keeps track of the number of other sequences (including the current one) having a sequence identity higher than a specified threshold.

    Parameters
    ----------
    pdb_codes: list
        Contains PDB identifiers.
    max_res_list_h: list
        Heavy chain residues of all data.
    max_res_list_l: list
        Light chain residues of all data.
    threshold: float
        Highest accepted sequence identity value.
    residues_path: str
        Path to the folder containing the list of residues per entry.

    """

    weights = np.zeros((len(pdb_codes)))

    for i, pdb_code in enumerate(pdb_codes):
        main_seq = get_sequence(list(np.load(residues_path+pdb_code+'.npy')), max_res_list_h=max_res_list_h, max_res_list_l=max_res_list_l)
        for pdb_code_ in pdb_codes:
            other_seq = get_sequence(list(np.load(residues_path+pdb_code_+'.npy')), max_res_list_h=max_res_list_h, max_res_list_l=max_res_list_l)
            identity = antibody_sequence_identity(main_seq, other_seq)
            if identity > threshold:
                weights[i] += 1

    return weights
'''
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

