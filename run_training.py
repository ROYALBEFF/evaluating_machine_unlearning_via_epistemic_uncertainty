import argparse

import mlflow
import torch.optim.optimizer
from time import time
import numpy as np
from utils import MetricProgressions, log_and_plot_progressions, log_artifact, log_metrics_and_artifacts, log_model, run_exists, log_batch_update
from data import build_batches, load_data
from model import MLP
from copy import deepcopy


def main():
    run_name = f'training_s_{args.s}_d_{args.d}_lr_{str(args.lr).replace(".", "_")}_e_{args.e}_bs_{args.bs}'
    if run_exists(run_name, args.o):
        raise RuntimeError(f'Run with name/configuration {run_name} already exists!')

    with mlflow.start_run(run_name=run_name):
        model_identifier = 'pre-trained'
        # load training and test data, download data first if necessary
        x_training, y_training, x_test, y_test = load_data(args.d)

        if args.d == 'mnist':
            input_dim = 784
        elif args.d == 'cifar10':
            input_dim = 1024
        else:
            raise ValueError(f'Unknown dataset {args.d}')

        # initialize torch and numpy rng
        torch.manual_seed(args.s)
        np.random.seed(args.s)

        model = MLP(input_dim)

        cross_entropy = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        # store batch indices and updates for amnesiac unlearning
        batch_indices_per_epoch = []
        batch_updates_per_epoch = []
        metric_progressions = MetricProgressions(10)
        training_time_progression = []

        model.train()
        for epoch in range(args.e):
            print(f'Epoch: {epoch}')
            batch_indices = build_batches(x_training.shape[0], args.bs)
            batch_indices_per_epoch.append(batch_indices)

            batch_updates = []
            total_time_epoch = 0
            for batch in batch_indices:
                x_batch = x_training[batch]
                y_batch = y_training[batch]

                # get model parameters before updating them in order to compute the difference afterwards
                prior_model_parameters = {name: deepcopy(parameter.detach().numpy()) for name, parameter in model.named_parameters()}

                start_time = time()
                prediction_batch = model(x_batch)
                loss = cross_entropy(prediction_batch, y_batch)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                end_time = time()

                total_time_epoch += end_time - start_time

                # get updated model parameters and compute batch update, which is the difference to the prior model parameters
                batch_update = {name: deepcopy(parameter.detach().numpy()) - prior_model_parameters[name] for name, parameter in model.named_parameters()}
                batch_updates.append(batch_update)

            batch_updates_per_epoch.append(batch_updates)
            mlflow.log_metric('Training time per epoch', total_time_epoch)
            training_time_progression.append(total_time_epoch)
            metrics_and_artifacts = log_metrics_and_artifacts(model, x_training, y_training, x_test, y_test, x_remaining=x_training, y_remaining=y_training, x_target=x_training, y_target=y_training,
                                                              model_identifier=model_identifier, num_classes=10, run_name=run_name, output=args.o, step=epoch, run_attack=False, compute_efficacy=False)
            metric_progressions.collect_metrics(metrics_and_artifacts)

        log_and_plot_progressions(metric_progressions, args.e, model_identifier, 10, run_name, args.o)
        log_artifact(np.array(training_time_progression), 'training_time', model_identifier, run_name, args.o)
        log_batch_update(batch_indices_per_epoch, batch_updates_per_epoch, run_name, args.o)
        log_model(model, model_identifier, run_name, args.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('-s', '-seed', type=int, required=True, help='Random seed.')
    parser.add_argument('-d', '-dataset', type=str, required=True, help='Dataset: mnist or cifar10.')
    parser.add_argument('-lr', '-learning-rate', type=float, required=True, help='SGD learning rate.')
    parser.add_argument('-e', '-epochs', type=int, required=True, help='Number of training epochs.')
    parser.add_argument('-bs', '-batch-size', type=int, required=True, help='Batch size.')
    parser.add_argument('-o', '-output', type=str, required=False, default='.', help='Output directory for artifacts, plots and models')
    args = parser.parse_args()
    main()
