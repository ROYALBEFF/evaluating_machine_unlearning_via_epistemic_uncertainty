import torch
import mlflow
import numpy as np
from forgetting import efficacy, efficacy_upper_bound


class MLP(torch.nn.Module):

    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        # model outputs logits!
        return x


class SmallMLP(torch.nn.Module):

    def __init__(self):
        super(SmallMLP, self).__init__()
        self.fc1 = torch.nn.Linear(2, 4)
        self.fc2 = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # model outputs logits!
        return x


class LogisticRegression(torch.nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        # model outputs logits!
        return self.linear(x)

    def get_decision_boundary(self):
        """Get decision boundary from -1 to 1"""
        parameters = list(self.parameters())
        w = parameters[0][0].detach()
        b = parameters[1][0].detach()
        return [float((-x * w[0] - b) / w[1]) for x in np.linspace(-1, 1, 10)]


def train(model, x, y, x_remaining, y_remaining, x_target, y_target, epochs, batch_size, learning_rate, model_identifier):
    """Train LogisticRegression or SmallMLP model."""
    model.train()
    labels = torch.argmax(y, 1) if len(y.shape) > 1 else y

    if isinstance(model, LogisticRegression):
        objective = torch.nn.MSELoss()
        # flag indicating if the model is a logistic regression model
        logistic_regression = True
        labels = labels.float()
    elif isinstance(model, SmallMLP):
        objective = torch.nn.CrossEntropyLoss()
        # flag indicating if the model is a logistic regression model
        logistic_regression = False
        labels = labels.long()
    else:
        raise ValueError(f'Unsupported model type {type(model)}')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # if batch size is -1, use whole dataset at once
    if batch_size == -1:
        batch_size = x.shape[0]

    for epoch in range(epochs):
        eff = efficacy(model, x_target, y_target, logistic_regression=logistic_regression)
        eff_upper_bound = efficacy_upper_bound(model, x_target, y_target, logistic_regression=logistic_regression)
        mlflow.log_metric(f'{model_identifier}_efficacy', eff, step=epoch)
        mlflow.log_metric(f'{model_identifier}_efficacy_upper_bound', eff_upper_bound, step=epoch)

        # permute data indices for batch learning
        permuted_indices = torch.randperm(x.shape[0])

        # train model on batches
        for i in range(0, x.shape[0], batch_size):
            batch_indices = permuted_indices[i:i + batch_size]
            output = model(x[batch_indices])
            if logistic_regression:
                output = torch.sigmoid(output).flatten()
            loss = objective(output, labels[batch_indices])
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # log training loss after each epoch
        output = model(x)
        if logistic_regression:
            output = torch.sigmoid(output).flatten()
        loss = objective(output, labels)
        mlflow.log_metric(f'{model_identifier}_training_loss', loss.tolist(), step=epoch)

    # log training loss after training
    output = model(x)
    if logistic_regression:
        output = torch.sigmoid(output).flatten()
    loss = objective(output, labels).tolist()
    mlflow.log_metric(f'{model_identifier}_training_loss', loss, step=epochs)

    eff = efficacy(model, x_target, y_target, logistic_regression=logistic_regression)
    eff_upper_bound = efficacy_upper_bound(model, x_target, y_target, logistic_regression=logistic_regression)
    mlflow.log_metric(f'{model_identifier}_efficacy', eff, step=epochs)
    mlflow.log_metric(f'{model_identifier}_efficacy_upper_bound', eff_upper_bound, step=epochs)
