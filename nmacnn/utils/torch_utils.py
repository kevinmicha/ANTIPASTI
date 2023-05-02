import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.autograd import Variable

def create_test_set(train_x, train_y, test_size=0.023):
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
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=test_size, random_state=9)

    # Converting to tensors
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[1])
    train_x = train_x.astype(np.float32)
    train_x  = torch.from_numpy(train_x)
    train_y = train_y.astype(np.float32).reshape(train_y.shape[0], 1)
    train_y = torch.from_numpy(train_y)

    test_x = test_x.reshape(test_x.shape[0], 1, train_x.shape[2], train_x.shape[2])
    test_x = test_x.astype(np.float32)
    test_x  = torch.from_numpy(test_x)
    test_y = test_y.astype(np.float32).reshape(test_y.shape[0], 1, 1)
    test_y = torch.from_numpy(test_y)

    return train_x, test_x, train_y, test_y

def training_step(model, criterion, optimiser, train_x, test_x, train_y, test_y, train_losses, test_losses, epoch, batch_size, verbose):
    r"""Performs a training step.
    
    Parameters
    ----------
    model: nmacnn.model.model.NormalModeAnalysisCNN
        The model class, i.e., ``NormalModeAnalysisCNN``.
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
    batch_size = batch_size

    x_train, y_train = Variable(train_x), Variable(train_y)
    x_test, y_test = Variable(test_x), Variable(test_y)

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
    loss_test = 0
    output_test, _ = model(x_test)
    for i in range(x_test.size()[0]):
        output_t, _ = model(x_test[i].reshape(1, 1, model.input_shape, model.input_shape))
        loss_t = criterion(output_t, y_test[i])
        loss_test += loss_t / x_test.size()[0]
        if verbose:
            print(output_t)
            print(y_test[i])
            print('------------------------')
    test_losses.append(loss_test)
    
    # Training and test losses
    print('Epoch : ', epoch+1, '\t', 'train loss: ', tr_loss, 'test loss :', loss_test)

        
    return train_losses, test_losses, inter_filter, y_test, output_test

def training_routine(model, criterion, optimiser, train_x, test_x, train_y, test_y, n_max_epochs=120, max_corr=0.87, batch_size=32, verbose=True):
    r"""Performs a chosen number of training steps.
    
    Parameters
    ----------
    model: nmacnn.model.model.NormalModeAnalysisCNN
        The model class, i.e., ``NormalModeAnalysisCNN``.
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
    train_losses = []
    test_losses = []

    for epoch in range(n_max_epochs):
        train_losses, test_losses, inter_filter, y_test, output_test = training_step(model, criterion, optimiser, train_x, test_x, train_y, test_y, train_losses, test_losses, epoch, batch_size, verbose)

        # Computing and printing the correlation coefficient
        corr = np.corrcoef(output_test.detach().numpy().T, y_test[:,0].detach().numpy().T)[1,0]
        if verbose:
            print('Corr: ' + str(corr))
        if corr > max_corr:
            break
    
    return train_losses, test_losses, inter_filter, y_test, output_test

    