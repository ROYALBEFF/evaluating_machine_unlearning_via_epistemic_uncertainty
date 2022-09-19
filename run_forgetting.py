import mlflow.pytorch
import argparse
import torch
import numpy as np
from time import time
from utils import *
from forgetting import *
from plot import *
from model import *
from data import *
import logging
import sys
import hashlib


def run_amnesiac_unlearning(model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, target_data_indices, history_path):
    time_start = time()
    model = amnesiac_unlearning(model, target_data_indices, history_path)
    time_end = time()

    # log time and other metrics and artifacts for the scrubbed model
    duration = time_end - time_start
    mlflow.log_metric('Forgetting time', duration)
    metrics_and_artifacts = log_metrics_and_artifacts(model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, 'amnesiac', 10, run_name, args.o,
                                                      attack_pre_trained=False, compute_efficacy=True)

    # plot membership inference probabilities
    plot_membership_inference_probabilities(metrics_and_artifacts['membership_inference_probabilities'], 'amnesiac', run_name, args.o)

    # plot data extracted by model inversion attack
    shape = get_2d_shape(x_training.shape[-1])
    x_extracted = metrics_and_artifacts['extracted_data']
    y_extracted = metrics_and_artifacts['extracted_data_labels']
    plot_model_inversion_extracted_data(x_extracted, y_extracted, shape, 'amnesiac', run_name, args.o)

    # log scrubbed model
    log_model(model, 'amnesiac', run_name, args.o)


def run_fisher_forgetting(model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, alpha):
    time_start = time()
    model = fisher_forgetting(model, x_remaining, y_remaining, alpha)
    time_end = time()

    # log time and other metrics and artifacts for the scrubbed model
    duration = time_end - time_start
    mlflow.log_metric('Forgetting time', duration)
    # log_artifact(duration, 'forgetting_time', 'fisher')
    metrics_and_artifacts = log_metrics_and_artifacts(model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, 'fisher', 10, run_name, args.o,
                                                      attack_pre_trained=False, compute_efficacy=True)

    # plot membership inference probabilities
    plot_membership_inference_probabilities(metrics_and_artifacts['membership_inference_probabilities'], 'fisher', run_name, args.o)

    # plot data extracted by model inversion attack
    shape = get_2d_shape(x_training.shape[-1])
    x_extracted = metrics_and_artifacts['extracted_data']
    y_extracted = metrics_and_artifacts['extracted_data_labels']
    plot_model_inversion_extracted_data(x_extracted, y_extracted, shape, 'fisher', run_name, args.o)

    # log scrubbed model
    log_model(model, 'fisher', run_name, args.o)


def run_retraining(x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, epochs, learning_rate, batch_size):
    time_accumulated = [0]
    metric_progressions = MetricProgressions(10)

    # model identifier for logs and plots
    model_identifier = 'retraining'

    # shape of data when transforming from one-dimensional to two-dimensional data
    num_input = x_training.shape[-1]

    logger.info(f'Reset random seeds to {args.s}!')
    # reset torch and numpy rng to guarantee identical model parameter initialization
    torch.manual_seed(args.s)
    np.random.seed(args.s)
    model = MLP(num_input)

    logger.info('Start model training!')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    cross_entropy = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch}...')
        batch_indices = build_batches(x_remaining.shape[0], batch_size)
        time_start = time()
        for batch in batch_indices:
            x_batch = x_remaining[batch]
            y_batch = y_remaining[batch]

            # perform foward and backward step
            prediction_batch = model(x_batch)
            loss = cross_entropy(prediction_batch, y_batch)
            model.zero_grad()
            loss.backward()
            optimizer.step()

        time_end = time()
        duration = time_end - time_start
        logger.info(f'Done in {duration} seconds!')
        # log time and other metrics and artifacts for the retrained model
        mlflow.log_metric('retraining_time', duration, epoch)

        logger.info(f'Log metrics and artifacts for epoch {epoch}...')
        is_last_epoch = epoch == epochs-1
        metrics_and_artifacts = log_metrics_and_artifacts(model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, model_identifier, 10, run_name, args.o, epoch,
                                                          run_attack=is_last_epoch, attack_pre_trained=False, compute_efficacy=is_last_epoch)

        # store overall and per-class accuracies, membership inference probability means and standard deviations and forgetting efficacy
        metric_progressions.collect_metrics(metrics_and_artifacts)

        # log accumulated time consumption for all epochs so far
        time_spend_so_far = time_accumulated[-1] + duration
        mlflow.log_metric('forgetting_time_accumulated', time_spend_so_far, epochs)
        time_accumulated.append(time_spend_so_far)
        logger.info('Done!')

    # plot membership inference probabilities
    plot_membership_inference_probabilities(metrics_and_artifacts['membership_inference_probabilities'], model_identifier, run_name, args.o)

    # log and plot progressions without gradient norm progression
    log_and_plot_progressions(metric_progressions, epochs, model_identifier, 10, run_name, args.o)

    # log accumulated times as artifact without the initial value 0
    log_artifact(np.array(time_accumulated[1:]), 'forgetting_time_accumulated', model_identifier, run_name, args.o)


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
                                                          run_attack=False, compute_efficacy=True)
        logger.info('Done!')

        model.train()
        if args.f.lower() == 'amnesiac':
            if args.hp == '':
                raise ValueError('History path argument must be provided for amnesiac unlearning!')
            run_amnesiac_unlearning(model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, target_indices, args.hp)
        elif args.f.lower() == 'fisher':
            if args.a == -1:
                raise ValueError('Alpha argument must be provided for Fisher forgetting!')
            run_fisher_forgetting(model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, args.a)
        elif args.f.lower() == 'retraining':
            if args.lr == -1:
                raise ValueError('Learning rate argument must be provided for retraining!')
            if args.e == -1:
                raise ValueError('Epochs argument must be provided for retraining!')
            if args.bs == -1:
                raise ValueError('Batch size argument must be provided for retraining!')
            logger.info('Start retraining!')
            run_retraining(x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, args.e, args.lr, args.bs)
        else:
            raise ValueError(f'Unknown forgetting method {args.f}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run forgetting experiment.')

    # General arguments
    parser.add_argument('-s', '-seed', type=int, required=True, help='Random seed.')
    parser.add_argument('-m', '-model', type=str, required=True, help='Path to pre-trained model.')
    parser.add_argument('-d', '-dataset', type=str, required=True, help='Dataset: mnist or cifar10.')
    parser.add_argument('-t', '-target', type=int, required=True, help='Target class to forget. -1 for mixed targets.')
    parser.add_argument('-p', '-percentage', type=float, required=True, help='Percentage of target data. Note: Percentage of whole training dataset for mixed targets.')
    parser.add_argument('-f', '-forgetting', type=str, required=True, help='Forgetting method. amnesiac, fisher or retraining.')
    parser.add_argument('-o', '-output', type=str, required=False, default='.', help='Output directory for artifacts, plots and models')

    # Amnesiac unlearning specific arguments
    parser.add_argument('-hp', '-history-path', type=str, required=False, default='', help='Amnesiac unlearning: Path to stored batch update history.')

    # Fisher forgetting specific arguments
    parser.add_argument('-a', '-alpha', type=float, required=False, default=-1, help='Fisher forgetting: Hyperparameter alpha combines lambda and sigma.')

    # Retraining specific arguments
    parser.add_argument('-lr', '-learning-rate', type=float, required=False, default=-1, help='Retraining: SGD learning rate.')
    parser.add_argument('-e', '-epochs', type=int, required=False, default=-1, help='Retraining: Number of retraining epochs.')
    parser.add_argument('-bs', '-batch-size', type=int, required=False, default=-1, help='Retraining: Batch size.')

    args = parser.parse_args()

    m = hashlib.sha1(args.m.encode("utf-8")).hexdigest()
    run_name = f's_{args.s}_d_{args.d}_t_{args.t}_p_{str(args.p).replace(".", "")}_f_{args.f}_m_{m}'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info('Run forgetting with parameters:')
    logger.info(f'Seed: {args.s}')
    logger.info(f'Model: {args.m}')
    logger.info(f'Dataset: {args.d}')
    logger.info(f'Target: {args.t}')
    logger.info(f'Percentage: {args.p}')
    logger.info(f'Forgetting: {args.f}')

    if args.f.lower() == 'amnesiac':
        logger.info(f'History path: {args.hp}')
    elif args.f.lower() == 'fisher':
        logger.info(f'Alpha: {args.a}')
        run_name += f'_a_{str(args.a).replace(".", "")}'
    elif args.f.lower() == 'retraining':
        logger.info(f'Learning rate: {args.lr}')
        logger.info(f'Epochs: {args.e}')
        logger.info(f'Batch size: {args.bs}')
        run_name += f'_lr_{str(args.lr).replace(".", "")}_e_{args.e}_bs_{args.bs}'

    if run_exists(run_name, args.o):
        raise RuntimeError(f'Run with name/configuration {run_name} already exists!')

    main()
