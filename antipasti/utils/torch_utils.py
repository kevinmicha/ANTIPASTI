import numpy as np
import torch

from adabelief_pytorch import AdaBelief
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from antipasti.model.model import ANTIPASTI

def create_test_set(train_x, train_y, test_size=None):
    r"""Creates the test set given a set of input images and their corresponding labels.

    Parameters
    ----------
    train_x: numpy.ndarray
        Input normal mode correlation maps.
    train_y: numpy.ndarray
        Labels.
    test_size: float
        Fraction of original samples to be included in the test set.

    Returns
    -------
    train_x: torch.Tensor
        Training inputs.
    test_x: torch.Tensor
        Test inputs.
    train_y: torch.Tensor
        Training labels. 
    test_y: torch.Tensor
        Test labels.

    """

    # Splitting
    indices = np.arange(len(train_x))
    if test_size == None:
        #indices_test = [44, 63, 94, 119, 144, 149, 254, 290, 328, 357, 402, 426, 464, 489, 604, 634]
        #indices_test = [44, 62, 91, 116, 141, 146, 250, 284, 322, 349, 389, 410, 447, 471, 574, 604]
        indices_test = [44, 62, 91, 116, 141, 146, 216, 250, 284, 348, 388, 409, 470, 573, 603, 648]
        np.random.shuffle(indices_test)
        indices_train = np.delete(indices, indices_test, axis=0)

        test_x = train_x[indices_test]
        test_y = train_y[indices_test]

        train_x = np.delete(train_x, indices_test, axis=0)
        train_y = np.delete(train_y, indices_test, axis=0)
    else:
        train_x, test_x, train_y, test_y, indices_train, indices_test = train_test_split(train_x, train_y, indices, test_size=test_size, random_state=9)

    # Converting to tensors
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[1])
    train_x = train_x.astype(np.float32)
    train_x  = torch.from_numpy(train_x)
    train_y = train_y.astype(np.float32).reshape(train_y.shape[0], 1)
    train_y = torch.from_numpy(train_y)

    test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[2], test_x.shape[2])
    test_x = test_x.astype(np.float32)
    test_x  = torch.from_numpy(test_x)
    test_y = test_y.astype(np.float32).reshape(test_y.shape[0], 1, 1)
    test_y = torch.from_numpy(test_y)

    return train_x, test_x, train_y, test_y, indices_train, indices_test

def training_step(model, criterion, optimiser, train_x, test_x, train_y, test_y, train_losses, test_losses, epoch, batch_size, verbose):
    r"""Performs a training step.
    
    Parameters
    ----------
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    criterion: torch.nn.modules.loss.MSELoss
        It calculates a gradient according to a selected loss function, i.e., ``MSELoss``.
    optimiser: adabelief_pytorch.AdaBelief.AdaBelief
        Method that implements an optimisation algorithm.
    train_x: torch.Tensor
        Training normal mode correlation maps.
    test_x: torch.Tensor
        Test normal mode correlation maps.
    train_y: torch.Tensor
        Training labels. 
    test_y: torch.Tensor
        Test labels.
    train_losses: list 
        The current history of training losses.
    test_losses: list 
        The current history of test losses.
    epoch: int
        Of value ``e`` if the dataset has gone through the model ``e`` times.
    batch_size: int
        Number of samples that pass through the model before its parameters are updated.
    verbose: bool
        ``True`` to print the losses in each epoch.
    
    Returns
    -------
    train_losses: list 
        The history of training losses after the training step.
    test_losses: list 
        The history of test losses after the training step.
    inter_filter: torch.Tensor
        Filters before the fully-connected layer.
    y_test: torch.Tensor
        Ground truth test labels.
    output_test: torch.Tensor
        The predicted test labels. 

    """   
    tr_loss = 0
    tr_mse = 0
    x_train, y_train = Variable(train_x), Variable(train_y)
    x_test, y_test = Variable(test_x), Variable(test_y)

    # Filters before the fully-connected layer
    size_inter = int(np.sqrt(model.fully_connected_input/model.n_filters))
    inter_filter = np.zeros((x_train.size()[0], model.n_filters, size_inter, size_inter))
    
    #perm_paired = []
    #perm_nano = []
    permutation = torch.randperm(x_train.size()[0])
    #for i in range(x_train.size()[0]):
    #    if torch.numel(torch.nonzero(x_train[i,0,-80:,-80:])) == 0:
    #        perm_nano.append(i)
    #    else:
    #        perm_paired.append(i)
    #np.random.shuffle(perm_nano)
    #np.random.shuffle(perm_paired)
    #print(len(perm_nano))
    #permutation = perm_nano + perm_paired

    for i in range(0, x_train.size()[0], batch_size):
        
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]
        
        # Training output
        output_train, inter_filters = model(batch_x)
        
        # Picking the appropriate filters before the fully-connected layer
        inter_filters_detached = inter_filters.detach().clone()
        inter_filter[i:i+batch_size] = inter_filters_detached.numpy()

        # Training loss, clearing gradients and updating weights
        #def closure():
        #    optimiser.zero_grad()
        #    output_train, inter_filters = model(batch_x)
        #    loss_train = criterion(output_train[:, 0], batch_y[:, 0])
        #    loss_train.backward()
        #    return loss_train

        #optimiser.step(closure)

        #with torch.no_grad():
        #    loss_train = criterion(output_train[:, 0], batch_y[:, 0]).detach()
        optimiser.zero_grad()
        l1_loss = model.l1_regularization_loss()
        mse_loss = criterion(output_train[:, 0], batch_y[:, 0])
        loss_train = mse_loss + l1_loss
        print(l1_loss)
        loss_train.backward()
        optimiser.step()
        # Adding batch contribution to training loss
        tr_loss += loss_train.item() * batch_size / x_train.size()[0]
        tr_mse += mse_loss * batch_size / x_train.size()[0]

    train_losses.append(tr_loss)
    loss_test = 0
    output_test = []
    for i in range(x_test.size()[0]):
        output_t, _ = model(x_test[i].reshape(1, 1, model.input_shape, model.input_shape))
        l1_loss = model.l1_regularization_loss()
        loss_t = criterion(output_t[:, 0], y_test[i][:, 0])
        loss_test += loss_t.item() / x_test.size()[0]
        if verbose:
            print(output_t)
            print(y_test[i])
            print('------------------------')
        output_test.append(output_t[:,0].detach().numpy())
    test_losses.append(loss_test)
    
    # Training and test losses
    print('Epoch : ', epoch+1, '\t', 'train loss: ', tr_loss, 'train MSE: ', tr_mse, 'test MSE: ', loss_test)

        
    return train_losses, test_losses, inter_filter, y_test, output_test

def training_routine(model, criterion, optimiser, train_x, test_x, train_y, test_y, n_max_epochs=120, max_corr=0.87, batch_size=32, verbose=True):
    r"""Performs a chosen number of training steps.
    
    Parameters
    ----------
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    criterion: torch.nn.modules.loss.MSELoss
        It calculates a gradient according to a selected loss function, i.e., ``MSELoss``.
    optimiser: adabelief_pytorch.AdaBelief.AdaBelief
        Method that implements an optimisation algorithm.
    train_x: torch.Tensor
        Training normal mode correlation maps.
    test_x: torch.Tensor
        Test normal mode correlation maps.
    train_y: torch.Tensor
        Training labels. 
    test_y: torch.Tensor
        Test labels.
    n_max_epochs: int
        Number of times the whole dataset goes through the model.
    max_corr: float
        If the correlation coefficient exceeds this value, the training routine is terminated.
    batch_size: int
        Number of samples that pass through the model before its parameters are updated.
    verbose: bool
        ``True`` to print the losses in each epoch.
    
    Returns
    -------
    train_losses: list 
        The history of training losses after the training routine.
    test_losses: list 
        The history of test losses after the training routine.
    inter_filter: torch.Tensor
        Filters before the fully-connected layer.
    y_test: torch.Tensor
        Ground truth test labels.
    output_test: torch.Tensor
        The predicted test labels. 

    """   
    train_losses = []
    test_losses = []

    for epoch in range(n_max_epochs):
        train_losses, test_losses, inter_filter, y_test, output_test = training_step(model, criterion, optimiser, train_x, test_x, train_y, test_y, train_losses, test_losses, epoch, batch_size, verbose)

        # Computing and printing the correlation coefficient
        corr = np.corrcoef(np.array(output_test).T, y_test[:,0].detach().numpy().T)[1,0]
        if verbose:
            print('Corr: ' + str(corr))
        if corr > max_corr:
            break
    
    return train_losses, test_losses, inter_filter, y_test, output_test

def load_checkpoint(path, input_shape, n_filters=None, pooling_size=None, filter_size=None):
    r"""Loads a checkpoint from the ``checkpoints`` folder.
    
    Parameters
    ----------
    path: str
        Checkpoint path.
    input_shape: int
        Shape of the normal mode correlation maps.
    n_filters: int
        Number of filters in the convolutional layer.
    filter_size: int
        Size of filters in the convolutional layer.
    pooling_size: int
        Size of the max pooling operation.
    
    Returns
    -------
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    optimiser: adabelief_pytorch.AdaBelief.AdaBelief
        Method that implements an optimisation algorithm.
    n_epochs: int
        Number of times the whole dataset went through the model.
    train_losses: list 
        The history of training losses after the training routine.
    test_losses: list 
        The history of test losses after the training routine.

    """  
    # Extracting parameters
    if path.endswith('_test.pt'):
        model = ANTIPASTI(input_shape=input_shape)
    else:
        if n_filters is None:
            nf = int(path.partition('_filters_')[-1][0])
            ps = int(path.partition('_pool_')[-1][0])
            fs = int(path.partition('_size_')[-1][0])
        else:
            nf = n_filters
            ps = pooling_size
            fs = filter_size
        model = ANTIPASTI(n_filters=nf, pooling_size=ps, filter_size=fs, input_shape=input_shape)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser = AdaBelief(model.parameters(), eps=1e-8, print_change_log=False) 
    if 'optimiser_state_dict' in checkpoint:
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    else:
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    n_epochs = checkpoint['epoch']
    train_losses = checkpoint['tr_loss']
    if 'test_loss' in checkpoint:
        test_losses = checkpoint['test_loss']
    else:
        test_losses = checkpoint['val_loss']

    return model, optimiser, n_epochs, train_losses, test_losses

def save_checkpoint(path, model, optimiser, train_losses, test_losses):
    r"""Saves a checkpoint in the ``checkpoints`` folder.
    
    Parameters
    ----------
    path: str
        Checkpoint path.
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    optimiser: adabelief_pytorch.AdaBelief.AdaBelief
        Method that implements an optimisation algorithm.
    train_losses: list 
        The history of training losses after the training routine.
    test_losses: list 
        The history of test losses after the training routine.

    """  
    EPOCH = len(test_losses)
    TR_LOSS = train_losses
    TEST_LOSS = test_losses

    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'tr_loss': TR_LOSS,
                'test_loss': TEST_LOSS,            
                }, path)
