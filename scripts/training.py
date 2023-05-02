import argparse
import numpy as np
import torch

from adabelief_pytorch import AdaBelief
from torch.nn import MSELoss

from nmacnn.model.model import NormalModeAnalysisCNN
from nmacnn.preprocessing.preprocessing import Preprocessing
from nmacnn.utils.torch_utils import create_validation_set, training_routine

args = None
parser = argparse.ArgumentParser(description='NarrowPSFGen Options')
parser.add_argument('--n_filters', dest='n_filters', type=int,
                    default=2, help='Number of filters in the convolutional layer.')
parser.add_argument('--filter_size', dest='filter_size', type=int,
                    default=5, help='Size of filters in the convolutional layer.')
parser.add_argument('--pooling_size', dest='pooling_size', type=int,
                    default=1, help='Size of the max pooling operation.')
parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                    default=4e-4, help='Step size at each iteration.')
parser.add_argument('--n_max_epochs', dest='n_max_epochs', type=int,
                    default=120, help='Number of times the whole dataset goes through the model.')
parser.add_argument('--max_corr', dest='max_corr', type=float,
                    default=0.87, help='If the correlation coefficient exceeds this value, the training routine is terminated.')
parser.add_argument('--batch_size', dest='batch_size', type=int,
                    default=32, help='Number of samples that pass through the model before its parameters are updated.')
arguments = parser.parse_args()

def main(args):
    chain_lengths_path = 'chain_lengths_paired/'
    dccm_map_path = 'dccm_maps_paired/'
    residues_path = 'lists_of_residues_paired/'
    pathological = ['5omm', '1mj7', '1qfw', '1qyg', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t']
    n_filters = args.n_filters
    filter_size = args.filter_size
    pooling_size = args.pooling_size
    learning_rate = args.learning_rate
    n_max_epochs = args.n_max_epochs
    max_corr = args.max_corr
    batch_size = args.batch_size

    # Preprocessing and creating the validatin set
    preprocessed_data = Preprocessing(chain_lengths_path=chain_lengths_path, dccm_map_path=dccm_map_path, residues_path=residues_path, pathological=pathological, renew_maps=False, renew_residues=False)
    train_x, val_x, train_y, val_y = create_validation_set(preprocessed_data.train_x, preprocessed_data.train_y, val_size=0.023)
    input_shape = preprocessed_data.train_x.shape[-1]
    
    # Defining the model, criterion and optimiser
    model = NormalModeAnalysisCNN(n_filters=n_filters, filter_size=filter_size, pooling_size=pooling_size, input_shape=input_shape)
    criterion = MSELoss()
    optimiser = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-8, print_change_log=False) 

    train_losses = []
    val_losses = []

    # Training
    train_loss, val_loss, inter_filter, y_val, output_val = training_routine(model, criterion, optimiser, train_x, val_x, train_y, val_y, n_max_epochs=n_max_epochs, max_corr=max_corr, batch_size=batch_size)

    # Printing prediction and ground truth (validation set)
    print(output_val)
    print(y_val)

    # Saving the losses
    train_losses.extend(train_loss)
    val_losses.extend(val_loss)

    ## Saving Neural Network checkpoint
    EPOCH = len(val_losses)
    PATH = 'model_' + n_max_epochs + '_epochs_' + pooling_size + '_pool_' + n_filters + '_filters.pt'
    TR_LOSS = train_losses
    VAL_LOSS = val_losses
    
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'tr_loss': TR_LOSS,
                'val_loss': VAL_LOSS,            
                }, PATH)

if __name__ == '__main__':
    main(arguments)
