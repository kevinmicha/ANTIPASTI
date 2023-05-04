import argparse
import numpy as np
import torch

from nmacnn.preprocessing.preprocessing import Preprocessing
from nmacnn.utils.torch_utils import load_checkpoint
from config import CHECKPOINTS_DIR

args = None
parser = argparse.ArgumentParser(description='Evaluation Options')
parser.add_argument('--n_filters', dest='n_filters', type=int,
                    default=2, help='Number of filters in the convolutional layer saved in the checkpoint.')
parser.add_argument('--filter_size', dest='filter_size', type=int,
                    default=5, help='Size of filters in the convolutional layer saved in the checkpoint.')
parser.add_argument('--pooling_size', dest='pooling_size', type=int,
                    default=1, help='Size of the max pooling operation saved in the checkpoint.')
parser.add_argument('--modes', dest='modes', type=int,
                    default=30, help='Normal modes into consideration when training.')
parser.add_argument('--n_max_epochs', dest='n_max_epochs', type=int,
                    default=120, help='Number of times the whole dataset went through the model when training.')
arguments = parser.parse_args()

def main(args):
    chain_lengths_path = 'chain_lengths_paired/'
    dccm_map_path = 'dccm_maps_paired/'
    residues_path = 'lists_of_residues_paired/'
    pathological = ['5omm', '1mj7', '1qfw', '1qyg', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t']
    stage = 'predicting'
    test_dccm_map_path = '../notebooks/test_data/dccm_map/'
    test_residues_path = '../notebooks/test_data/list_of_residues/'
    test_structure_path = '../notebooks/test_data/structure/'
    n_filters = args.n_filters
    filter_size = args.filter_size
    pooling_size = args.pooling_size
    modes = args.modes
    n_max_epochs = args.n_max_epochs

    # Loading a test sample
    preprocessed_data = Preprocessing(chain_lengths_path='chain_lengths_paired/', dccm_map_path='dccm_maps_paired/', residues_path='lists_of_residues_paired/', modes=modes, pathological=pathological, stage=stage, test_dccm_map_path=test_dccm_map_path, test_residues_path=test_residues_path, test_structure_path=test_structure_path)
    input_shape = preprocessed_data.test_x.shape[-1]
    
    # Loading an NMA-CNN checkpoint
    path = CHECKPOINTS_DIR + 'model_epochs_' + str(n_max_epochs) + '_modes_' + str(modes) + '_pool_' + str(pooling_size) + '_filters_' + str(n_filters) + '_size_' + str(filter_size) + '.pt'
    model = load_checkpoint(path, input_shape)[0]
    model.eval()

    # Predicting the binding affinity
    test_sample = torch.from_numpy(preprocessed_data.test_x.reshape(1, 1, input_shape, input_shape).astype(np.float32))

    print('The output value is ' + str(model(test_sample)[0].detach().numpy()[0,0]))
    print('So the binding affinity is ' + str(10**model(test_sample)[0].detach().numpy()[0,0]))


if __name__ == '__main__':
    main(arguments)
