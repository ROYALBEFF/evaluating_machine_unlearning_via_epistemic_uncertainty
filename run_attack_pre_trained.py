import mlflow.pytorch
import argparse
import torch
import numpy as np
from time import time
from utils import *
from plot import *
from model import *
from data import *
import logging
import sys
import hashlib


def main():
    with mlflow.start_run(run_name=run_name):
        # initialize torch and numpy rng
        torch.manual_seed(args.s)
        np.random.seed(args.s)

        logger.info('Load model...')
        # load pre-trained model
        with open(args.m, 'rb') as model_file:
            model = torch.load(model_file)
        logger.info('Done!')

        # model identifier for logs and plots
        model_identifier = 'pre-trained'

        logger.info('Prepare dataset...')
        # load training and test data, download data first if necessary
        x_training, y_training, x_test, y_test = load_data(args.d)
        # get target-remaining split of training data and the indices of the data points
        x_target, y_target, target_indices, x_remaining, y_remaining, remaining_indices = split_into_remaining_and_target_data(x_training, y_training, args.t, args.p)
        # log data set split as well as metric and artifacts for the pre-trained model
        log_artifact(target_indices.numpy(), 'target_data_indices', '', run_name, args.o)
        log_artifact(remaining_indices.numpy(), 'remaining_data_indices', '', run_name, args.o)
        logger.info('Done!')

        logger.info('Log metrics and artifacts for the pre-trained model...')
        metrics_and_artifacts = log_metrics_and_artifacts(model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, model_identifier, 10, run_name, args.o,
                                                          attack_pre_trained=True, compute_efficacy=True)
        logger.info('Done!')

        # plot membership inference probabilities
        plot_membership_inference_probabilities(metrics_and_artifacts['membership_inference_probabilities'], model_identifier, run_name, args.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack pre-trained model.')
    parser.add_argument('-s', '-seed', type=int, required=True, help='Random seed.')
    parser.add_argument('-m', '-model', type=str, required=True, help='Path to pre-trained model.')
    parser.add_argument('-d', '-dataset', type=str, required=True, help='Dataset: mnist or cifar10.')
    parser.add_argument('-t', '-target', type=int, required=True, help='Target class to forget. -1 for mixed targets.')
    parser.add_argument('-p', '-percentage', type=float, required=True, help='Percentage of target data. Note: Percentage of whole training dataset for mixed targets.')
    parser.add_argument('-o', '-output', type=str, required=False, default='.', help='Output directory for artifacts, plots and models')
    args = parser.parse_args()

    m = hashlib.sha1(args.m.encode("utf-8")).hexdigest()
    run_name = f'attack_s_{args.s}_d_{args.d}_t_{args.t}_p_{str(args.p).replace(".", "")}_m_{m}'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info('Attack pre-trained model with parameters:')
    logger.info(f'Seed: {args.s}')
    logger.info(f'Model: {args.m}')
    logger.info(f'Dataset: {args.d}')
    logger.info(f'Target: {args.t}')
    logger.info(f'Percentage: {args.p}')

    if run_exists(run_name, args.o):
        raise RuntimeError(f'Run with name/configuration {run_name} already exists!')

    main()
