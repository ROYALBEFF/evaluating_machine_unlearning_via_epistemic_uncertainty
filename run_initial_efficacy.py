import argparse

import mlflow
import torch.optim.optimizer
import numpy as np
from utils import log_metrics_and_artifacts, run_exists
from data import load_data, split_into_remaining_and_target_data
from model import MLP


def main():
    run_name = f'init_s_{args.s}_d_{args.d}_t_{args.t}_p_{args.p}'
    if run_exists(run_name, args.o):
        raise RuntimeError(f'Run with name/configuration {run_name} already exists!')

    with mlflow.start_run(run_name=run_name):
        model_identifier = 'init'

        # load training and test data, download data first if necessary
        x_training, y_training, x_test, y_test = load_data(args.d)
        # get target-remaining split of training data and the indices of the data points
        x_target, y_target, target_indices, x_remaining, y_remaining, remaining_indices = split_into_remaining_and_target_data(x_training, y_training, args.t, args.p)

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
        log_metrics_and_artifacts(model, x_training, y_training, x_test, y_test, x_remaining=x_remaining, y_remaining=y_remaining, x_target=x_target, y_target=y_target,
                                  model_identifier=model_identifier, num_classes=10, run_name=run_name, output=args.o, step=None, run_attack=False, compute_efficacy=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute efficacy for initialized model.')
    parser.add_argument('-s', '-seed', type=int, required=True, help='Random seed.')
    parser.add_argument('-d', '-dataset', type=str, required=True, help='Dataset: mnist or cifar10.')
    parser.add_argument('-t', '-target', type=int, required=True, help='Target class to forget. -1 for mixed targets.')
    parser.add_argument('-p', '-percentage', type=float, required=True, help='Percentage of target data. Note: Percentage of whole training dataset for mixed targets.')
    parser.add_argument('-o', '-output', type=str, required=False, default='.', help='Output directory for artifacts, plots and models')
    args = parser.parse_args()
    main()
