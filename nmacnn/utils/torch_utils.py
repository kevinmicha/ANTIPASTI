import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.autograd import Variable

def create_validation_set(train_x, train_y, val_size=0.023):
    """
    Creates the validation set given a set of input images and their corresponding labels.

    :param train_x: array of input normal mode correlation maps
    :param train_y: array of labels
    :param val_size: fraction of original samples to be included in the validation set
    :type val_size: float

    :return: four tensors. The order is the following: training inputs, validations inputs, training labels and validation labels

    """

    # Splitting
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_size, random_state=9)

    # Converting to tensors
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[1])
    train_x = train_x.astype(np.float32)
    train_x  = torch.from_numpy(train_x)
    train_y = train_y.astype(np.float32).reshape(train_y.shape[0], 1)
    train_y = torch.from_numpy(train_y)

    val_x = val_x.reshape(val_x.shape[0], 1, train_x.shape[2], train_x.shape[2])
    val_x = val_x.astype(np.float32)
    val_x  = torch.from_numpy(val_x)
    val_y = val_y.astype(np.float32).reshape(val_y.shape[0], 1, 1)
    val_y = torch.from_numpy(val_y)

    return train_x, val_x, train_y, val_y

def training_step(model, criterion, optimiser, train_x, val_x, train_y, val_y, train_losses, val_losses, epoch, batch_size, verbose):
    """
    Performs a training step.

    :param model: Machine Learning model, for example, ``NormalModeAnalysisCNN``
    :param criterion: calculates a gradient according to a selected loss function
    :param optimiser: method that implements an optimisation algorithm
    :param train_x: array of training normal mode correlation maps
    :param val_x: array of validation normal mode correlation maps
    :param train_y: array of training labels
    :param val_y: array of validation labels
    :param train_losses: list containing the history of training losses
    :param val_losses: list containing the history of validation losses
    :param epoch: of value ``e`` if the dataset has gone through the model ``e`` times
    :type epoch: int
    :param batch_size: number of samples that pass through the model before its parameters are updated
    :type batch_size: int
    :param verbose: ``True`` to print the losses in each epoch
    :type verbose: bool

    """   
    tr_loss = 0
    batch_size = batch_size

    x_train, y_train = Variable(train_x), Variable(train_y)
    x_val, y_val = Variable(val_x), Variable(val_y)

    # Filters before the fully-connected layer
    size_inter = int(np.sqrt(model.fully_connected_input/model.n_filters))
    inter_filter = np.zeros((x_train.size()[0], model.n_filters, size_inter, size_inter))
    
    permutation = torch.randperm(x_train.size()[0])

    for i in range(0, x_train.size()[0], batch_size):
        
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]
        
        # Training output
        output_train, inter_filters = model(batch_x)
        
        # Picking the appropriate filters before the fully-connected layer
        inter_filter[i:i+batch_size] = inter_filters.detach().numpy()

        # Training loss, clearing gradients and updating weights
        loss_train = criterion(output_train, batch_y)
        optimiser.zero_grad()
        loss_train.backward()
        optimiser.step()    
        
        # Adding batch contribution to training loss
        tr_loss += loss_train.item() * batch_size / x_train.size()[0]

    train_losses.append(tr_loss)
    loss_val = 0
    output_val, _ = model(x_val)
    for i in range(x_val.size()[0]):
        output_v, _ = model(x_val[i].reshape(1, 1, model.input_shape, model.input_shape))
        loss_v = criterion(output_v, y_val[i])
        loss_val += loss_v / x_val.size()[0]
        if verbose:
            print(output_v)
            print(y_val[i])
            print('------------------------')
    val_losses.append(loss_val)
    
    # Training and validation losses
    print('Epoch : ', epoch+1, '\t', 'train loss: ', tr_loss, 'val loss :', loss_val)

        
    return train_losses, val_losses, inter_filter, y_val, output_val

def training_routine(model, criterion, optimiser, train_x, val_x, train_y, val_y, n_max_epochs=120, max_corr=0.87, batch_size=32, verbose=True):
    """
    Performs a chosen number of training steps.

    :param model: Machine Learning model, for example, ``NormalModeAnalysisCNN``
    :param criterion: calculates a gradient according to a selected loss function
    :param optimiser: method that implements an optimisation algorithm
    :param train_x: array of training normal mode correlation maps
    :param val_x: array of validation normal mode correlation maps
    :param train_y: array of training labels
    :param val_y: array of validation labels
    :param n_max_epochs: number of times the whole dataset goes through the model
    :type n_max_epochs: int
    :param max_corr: if the correlation coefficient exceeds this value, the training routine is terminated
    :type max_corr: float
    :param batch_size: number of samples that pass through the model before its parameters are updated
    :type batch_size: int
    :param verbose: ``True`` to print the losses in each epoch
    :type verbose: bool

    """    
    train_losses = []
    val_losses = []

    for epoch in range(n_max_epochs):
        train_losses, val_losses, inter_filter, y_val, output_val = training_step(model, criterion, optimiser, train_x, val_x, train_y, val_y, train_losses, val_losses, epoch, batch_size, verbose)

        # Computing and printing the correlation coefficient
        corr = np.corrcoef(output_val.detach().numpy().T, y_val[:,0].detach().numpy().T)[1,0]
        if verbose:
            print('Corr: ' + str(corr))
        if corr > max_corr:
            break
    
    return train_losses, val_losses, inter_filter, y_val, output_val

    