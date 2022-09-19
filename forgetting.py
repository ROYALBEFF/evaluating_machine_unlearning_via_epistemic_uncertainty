import torch
from torch.autograd import grad
import pickle
import numpy as np
from time import time
from attack import onehot
from tqdm import tqdm


def contains_any(search_in, look_for):
    """Check if search_in contains any element in look_for by checking if their intersection is empty."""
    intersection = np.intersect1d(search_in, look_for)
    return intersection.size > 0


def efficacy(model, x, y, logistic_regression=False):
    """Return forgetting score (efficacy)."""
    information_target_data = information_score(model, x, y, logistic_regression=logistic_regression)
    eff = torch.inf if information_target_data == 0 else 1. / information_target_data
    return eff.tolist()


def efficacy_upper_bound(model, x, y, logistic_regression=False):
    """Return upper bound for forgetting score (efficacy)."""
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    predictions = model(x)
    if logistic_regression:
        predictions = torch.concat([-predictions, predictions], dim=-1)
    loss = ce_loss(predictions, y)
    model.zero_grad()
    loss.backward()
    squared_norm = gradient_norm(model) ** 2
    return torch.inf if squared_norm == 0 else 1. / squared_norm


def information_score(model, x, y, training=False, logistic_regression=False):
    """
    Compute Fisher-based information score for the given model and data.
    The training argument determines if the resulting tensor requires grad and also if the computational graph should be created for the gradient.
    """
    # get model prediction
    if logistic_regression:
        output = model(x)
        probabilities = torch.sigmoid(output)
        # create tensor of output probabilities for both classes
        predictions = torch.log(torch.concat([1-probabilities, probabilities], dim=-1))
    else:
        predictions = torch.log_softmax(model(x), dim=-1)

    # if x is just a single data point, expand the tensor by an additional dimension, such that x.shape = [1, n], expand y and predictions accordingly
    y = y if len(x.shape) > 1 else y[None, :]
    # guarantee that y is one-hot encoded
    y = y if len(y.shape) > 1 and not logistic_regression else torch.tensor(onehot(y, 10))
    predictions = predictions if len(x.shape) > 1 else predictions[None, :]
    x = x if len(x.shape) > 1 else x[None, :]
    num_data_points = x.shape[0]

    information = torch.tensor([0.], requires_grad=training)
    # accumulate information score for all data points
    for i in range(num_data_points):
        label = torch.argmax(y[i])
        prediction = predictions[i][label]
        # gradient of model prediction w.r.t. the model parameters
        gradient = grad(prediction, model.parameters(), create_graph=training, retain_graph=True)
        for derivative in gradient:
            information = information + torch.sum(derivative**2)

    # "convert" single-valued tensor to float value
    information = information[0]
    # return averaged information score
    return information / num_data_points


def approximate_fisher_information_matrix(model, x, y):
    """Levenberg-Marquart approximation of the Fisher information matrix diagonal."""
    # get model prediction
    predictions = torch.log_softmax(model(x), dim=-1)

    # if x is just a single data point, expand the tensor by an additional dimension, such that x.shape = [1, n], expand y accordingly
    y = y if len(x.shape) > 1 else y[None, :]
    x = x if len(x.shape) > 1 else x[None, :]
    num_data_points = x.shape[0]

    # initialize fisher approximation with 0 for each model parameter
    fisher_approximation = []
    for parameter in model.parameters():
        fisher_approximation.append(torch.zeros_like(parameter))

    epsilon = 10e-8
    # accumulate fisher approximation for all data points
    model.train()
    for i in tqdm(range(num_data_points)):
        label = torch.argmax(y[i])
        prediction = predictions[i][label]
        # gradient of model prediction w.r.t. the model parameters
        gradient = grad(prediction, model.parameters(), retain_graph=True, create_graph=False)
        for j, derivative in enumerate(gradient):
            # add a small constant epsilon to prevent dividing by 0 later
            fisher_approximation[j] += (derivative + epsilon)**2

    return fisher_approximation


def fisher_forgetting(model, x_remaining, y_remaining, alpha):
    """Perform Fisher forgetting as presented in Golatkar et al. 2020."""
    # approximate Fisher information matrix diagonal
    fisher_approximation = approximate_fisher_information_matrix(model, x_remaining, y_remaining)

    for i, parameter in enumerate(model.parameters()):
        # clamping the approximated fisher values according to the implementation details of Golatkar et al.
        noise = torch.sqrt(alpha / fisher_approximation[i]).clamp(max=1e-3) * torch.empty_like(parameter).normal_(0, 1)
        # increasing the noise of the last layer according to the implementation details of Golatkat et al.
        noise = noise * 10 if parameter.shape[-1] == 10 else noise
        parameter.data = parameter.data + noise

    return model


def amnesiac_unlearning(model, target_data_indices, history_path):
    """Perform amnesiac unlearning as presented in Graves et al. 2020."""
    # load history of batch updates during the training
    with open(history_path, 'rb') as history_file:
        batches, updates = pickle.load(history_file)

    # find sensitive batches and corresponding updates
    sensitive_batches = [(epoch_index, batch_index) for epoch_index, epoch in enumerate(batches) for batch_index, batch in enumerate(epoch) if contains_any(batch, target_data_indices)]
    sensitive_updates = [updates[epoch][batch] for epoch, batch in sensitive_batches]
    # accumulate sensitive batch updates for each parameter
    sensitive_updates = {name: np.stack([update[name] for update in sensitive_updates]).sum(axis=0) for name, parameter in model.named_parameters()}

    # remove sensitive updates from model parameter
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data - sensitive_updates[name]

    return model


def gradient_norm(model):
    """Compute norm of gradient vector w.r.t. the model parameters."""
    gradient = torch.concat([p.grad.data.flatten() for p in model.parameters()])
    norm = torch.linalg.norm(gradient).tolist()
    return norm

