import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from plot import plot_accuracy_progression, plot_membership_inference_probability_mean_and_std_progression, plot_gradient_norm_progression, plot_efficacy_progression
from attack import membership_inference_attack, model_inversion_attack
from forgetting import efficacy, efficacy_upper_bound
import mlflow.pytorch
import mlflow
import pathlib
from time import time


class MetricProgressions:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.efficacy_progression = []
        self.efficacy_upper_bound_progression = []
        self.overall_test_accuracy_progression = []
        self.overall_training_accuracy_progression = []
        self.overall_remaining_accuracy_progression = []
        self.overall_target_accuracy_progression = []
        self.class_test_accuracy_progressions = [[] for _ in range(num_classes)]
        self.class_training_accuracy_progressions = [[] for _ in range(num_classes)]
        self.class_remaining_accuracy_progressions = [[] for _ in range(num_classes)]
        self.class_target_accuracy_progressions = [[] for _ in range(num_classes)]
        self.membership_inference_probability_mean_progression = []
        self.membership_inference_probability_std_progression = []
        self.gradient_norm_progression = []

    def collect_metrics(self, metrics_and_artifacts):
        # store overall and per class accuracies for later use
        self.overall_test_accuracy_progression.append(metrics_and_artifacts['overall_test_accuracy'])
        self.overall_training_accuracy_progression.append(metrics_and_artifacts['overall_training_accuracy'])
        self.overall_remaining_accuracy_progression.append(metrics_and_artifacts['overall_remaining_accuracy'])
        self.overall_target_accuracy_progression.append(metrics_and_artifacts['overall_target_accuracy'])

        for label in range(self.num_classes):
            self.class_test_accuracy_progressions[label].append(metrics_and_artifacts['class_test_accuracies'][label])
            self.class_training_accuracy_progressions[label].append(metrics_and_artifacts['class_training_accuracies'][label])
            self.class_remaining_accuracy_progressions[label].append(metrics_and_artifacts['class_remaining_accuracies'][label])
            self.class_target_accuracy_progressions[label].append(metrics_and_artifacts['class_target_accuracies'][label])

        self.efficacy_progression.append(metrics_and_artifacts['efficacy'])
        self.efficacy_upper_bound_progression.append(metrics_and_artifacts['efficacy_upper_bound'])

        # store membership inference probability means and standard deviations for later use
        self.membership_inference_probability_mean_progression.append(metrics_and_artifacts['membership_inference_probability_mean'])
        self.membership_inference_probability_std_progression.append(metrics_and_artifacts['membership_inference_probability_std'])


def run_exists(run_name, output):
    """Check if a run with the given run configuration already exists.
    :param output:
    """
    artifacts_path = pathlib.Path(f'{output}/artifacts/{run_name}/')
    plots_path = pathlib.Path(f'{output}/plots/{run_name}/')
    models_path = pathlib.Path(f'{output}/models/{run_name}/')
    return artifacts_path.exists() or plots_path.exists() or models_path.exists()


def get_2d_shape(num_input):
    """Get 2D shape of one-dimensional image vectors"""
    return [int(np.sqrt(num_input))] * 2


def get_model_prefix(run_name, output):
    model_prefix = f'{output}/models/{run_name}/'
    # create directory if necessary
    pathlib.Path(model_prefix).mkdir(parents=True, exist_ok=True)
    return model_prefix


def get_artifact_prefix(run_name, output):
    artifact_prefix = f'{output}/artifacts/{run_name}/'
    # create directory if necessary
    pathlib.Path(artifact_prefix).mkdir(parents=True, exist_ok=True)
    return artifact_prefix


def get_batch_update_prefix(run_name, output):
    batch_update_prefix = f'{output}/batch_updates/{run_name}/'
    # create directory if necessary
    pathlib.Path(batch_update_prefix).mkdir(parents=True, exist_ok=True)
    return batch_update_prefix


def evaluate_accuracy(model, x_test, y_test, num_classes, logistic_regression=False):
    """Compute the overall model accuracy as well as the accuracy for each class."""
    # get model's predictions for the test dataset
    if logistic_regression:
        # in case of logistic regression compute sigmoid instead of log softmax
        probabilities = torch.sigmoid(model(x_test))
        y_pred = (probabilities >= 0.5).long().flatten().detach().numpy()
    else:
        y_pred = torch.argmax(torch.log_softmax(model(x_test), dim=1), dim=1).detach().numpy()

    # get accuracy per class
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    class_accuracies = [0] * num_classes
    for label in range(num_classes):
        # the class accuracy are the number of correct predictions of that class (TP) divided by the total number of predictions of that class (TP + FP)
        class_accuracies[label] = cm[label, label] / np.sum(cm[label, :]) if np.sum(cm[label, :]) > 0 else 0

    # get accuracy over all classes
    overall_accuracy = np.trace(cm) / len(y_test)

    return overall_accuracy, class_accuracies


def log_model(model, model_identifier, run_name, output, step=None):
    # define model path
    model_name = f'{model_identifier}'
    model_name += '' if step is None else f'_{step}'
    model_name += '.pth'
    model_path = get_model_prefix(run_name, output) + model_name

    # dump model as pickle and log model
    with open(model_path, 'wb+') as model_file:
        torch.save(model, model_path)
    mlflow.pytorch.log_model(model, f'models/{model_name}')


def log_batch_update(indices, updates, run_name, output):
    batch_update_path = get_batch_update_prefix(run_name, output) + 'batch_updates.pkl'

    # dump object as pickle and log artifact
    with open(batch_update_path, 'wb+') as batch_update_file:
        pickle.dump((indices, updates), batch_update_file)


def log_artifact(obj, metric_identifier, model_identifier, run_name, output, step=None):
    """Pickle object and log pickle file as MLflow artifact.
    :param output:
    :param run_name:
    """
    # define artifact path
    artifact_name = f'{metric_identifier}_{model_identifier}'
    artifact_name += '' if step is None else f'_{step}'
    artifact_name += '.pkl'
    artifact_path = get_artifact_prefix(run_name, output) + artifact_name

    # dump object as pickle and log artifact
    with open(artifact_path, 'wb+') as artifact_file:
        pickle.dump(obj, artifact_file)


def log_overall_and_class_accuracies(metrics_and_artifacts, model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, model_identifier, num_classes, run_name,
                                     output, step=None, logistic_regression=False):
    # overall and per-class accuracy
    overall_test_accuracy, class_test_accuracies = evaluate_accuracy(model, x_test, y_test, num_classes, logistic_regression=logistic_regression)
    overall_training_accuracy, class_training_accuracies = evaluate_accuracy(model, x_training, y_training, num_classes, logistic_regression=logistic_regression)
    overall_remaining_accuracy, class_remaining_accuracies = evaluate_accuracy(model, x_remaining, y_remaining, num_classes, logistic_regression=logistic_regression)
    overall_target_accuracy, class_target_accuracies = evaluate_accuracy(model, x_target, y_target, num_classes, logistic_regression=logistic_regression)

    metrics_and_artifacts['overall_test_accuracy'] = overall_test_accuracy
    metrics_and_artifacts['overall_training_accuracy'] = overall_training_accuracy
    metrics_and_artifacts['overall_remaining_accuracy'] = overall_remaining_accuracy
    metrics_and_artifacts['overall_target_accuracy'] = overall_target_accuracy
    metrics_and_artifacts['class_test_accuracies'] = class_test_accuracies
    metrics_and_artifacts['class_training_accuracies'] = class_training_accuracies
    metrics_and_artifacts['class_remaining_accuracies'] = class_remaining_accuracies
    metrics_and_artifacts['class_target_accuracies'] = class_target_accuracies

    mlflow.log_metric(f'{model_identifier}_overall_test_accuracy', overall_test_accuracy, step)
    mlflow.log_metric(f'{model_identifier}_overall_training_accuracy', overall_training_accuracy, step)
    mlflow.log_metric(f'{model_identifier}_overall_remaining_accuracy', overall_remaining_accuracy, step)
    mlflow.log_metric(f'{model_identifier}_overall_target_accuracy', overall_target_accuracy, step)

    for label in range(num_classes):
        mlflow.log_metric(f'{model_identifier}_class_{label}_test_accuracy', class_test_accuracies[label], step)
        mlflow.log_metric(f'{model_identifier}_class_{label}_training_accuracy', class_training_accuracies[label], step)
        mlflow.log_metric(f'{model_identifier}_class_{label}_remaining_accuracy', class_remaining_accuracies[label], step)
        mlflow.log_metric(f'{model_identifier}_class_{label}_target_accuracy', class_target_accuracies[label], step)

    log_artifact((overall_test_accuracy, class_test_accuracies), 'test_accuracy', model_identifier, run_name, output, step)
    log_artifact((overall_training_accuracy, class_training_accuracies), 'training_accuracy', model_identifier, run_name, output, step)
    log_artifact((overall_remaining_accuracy, class_remaining_accuracies), 'remaining_accuracy', model_identifier, run_name, output, step)
    log_artifact((overall_target_accuracy, class_target_accuracies), 'target_accuracy', model_identifier, run_name, output, step)

    return metrics_and_artifacts


def log_membership_inference_and_model_inversion(metrics_and_artifacts, model, x_training, y_training, x_test, y_test, x_target, y_target, model_identifier, run_name, output, step=None):
    # perform membership inference attack
    membership_inference_probabilities, attack_model = membership_inference_attack(model, x_training, y_training, x_test, y_test, x_target, y_target)

    log_artifact(membership_inference_probabilities, 'membership_inference_probabilities', model_identifier, run_name, output, step)
    metrics_and_artifacts['membership_inference_probabilities'] = membership_inference_probabilities.flatten()

    # log mean of membership inference probabilities
    mean = np.mean(membership_inference_probabilities)
    mlflow.log_metric(f'{model_identifier}_membership_inference_probability_mean', float(mean), step)
    # log_artifact(mean, 'membership_inference_probability_mean', model_identifier, step)
    metrics_and_artifacts['membership_inference_probability_mean'] = mean

    # log standard deviation of membership inference probabilities
    std = np.std(membership_inference_probabilities)
    mlflow.log_metric(f'{model_identifier}_membership_inference_probability_std', float(std), step)
    metrics_and_artifacts['membership_inference_probability_std'] = std

    metrics_and_artifacts['extracted_data'] = []
    metrics_and_artifacts['extracted_data_labels'] = []

    return metrics_and_artifacts


def log_efficacy_and_upper_bound(metrics_and_artifacts, model, x_target, y_target, model_identifier, run_name, output, logistic_regression=False):
    eff = efficacy(model, x_target, y_target, logistic_regression=logistic_regression)
    eff_upper_bound = efficacy_upper_bound(model, x_target, y_target, logistic_regression=logistic_regression)

    mlflow.log_metric(f'{model_identifier}_efficacy', eff)
    mlflow.log_metric(f'{model_identifier}_efficacy_upper_bound', eff_upper_bound)

    log_artifact(eff, 'efficacy', model_identifier, run_name, output)
    log_artifact(eff_upper_bound, 'efficacy_upper_bound', model_identifier, run_name, output)

    metrics_and_artifacts['efficacy'] = eff
    metrics_and_artifacts['efficacy_upper_bound'] = eff_upper_bound

    return metrics_and_artifacts


def log_and_plot_progressions(metric_progressions, steps, model_identifier, num_classes, run_name, output):
    # log and plot overall and class accuracy progressions
    log_artifact(np.array(metric_progressions.overall_test_accuracy_progression), 'overall_test_accuracy_progression', model_identifier, run_name, output)
    log_artifact(np.array(metric_progressions.overall_training_accuracy_progression), 'overall_training_accuracy_progression', model_identifier, run_name, output)
    log_artifact(np.array(metric_progressions.overall_remaining_accuracy_progression), 'overall_remaining_accuracy_progression', model_identifier, run_name, output)
    log_artifact(np.array(metric_progressions.overall_target_accuracy_progression), 'overall_target_accuracy_progression', model_identifier, run_name, output)
    for label in range(num_classes):
        log_artifact(np.array(metric_progressions.class_test_accuracy_progressions[label]), f'class_{label}_test_accuracy_progressions', model_identifier, run_name, output)
        log_artifact(np.array(metric_progressions.class_training_accuracy_progressions[label]), f'class_{label}_training_accuracy_progressions', model_identifier, run_name, output)
        log_artifact(np.array(metric_progressions.class_remaining_accuracy_progressions[label]), f'class_{label}_remaining_accuracy_progressions', model_identifier, run_name, output)
        log_artifact(np.array(metric_progressions.class_target_accuracy_progressions[label]), f'class_{label}_target_accuracy_progressions', model_identifier, run_name, output)
    plot_accuracy_progression(metric_progressions, steps, num_classes, model_identifier, run_name, output)

    # log and plot membership inference mean and standard deviation progressions
    log_artifact(np.array(metric_progressions.membership_inference_probability_mean_progression), 'membership_inference_probability_mean_progression', model_identifier, run_name, output)
    log_artifact(np.array(metric_progressions.membership_inference_probability_std_progression), 'membership_inference_probability_std_progression', model_identifier, run_name, output)

    # log and plot efficacy and efficacy upper bound
    log_artifact(np.array(metric_progressions.efficacy_progression), 'efficacy_progression', model_identifier, run_name, output)
    log_artifact(np.array(metric_progressions.efficacy_upper_bound_progression), 'efficacy_upper_bound_progression', model_identifier, run_name, output)

    # log and plot gradient norm progression
    if metric_progressions.gradient_norm_progression:
        log_artifact(np.array(metric_progressions.gradient_norm_progression), 'gradient_norm_progression', model_identifier, run_name, output)
        plot_gradient_norm_progression(metric_progressions, run_name, output)


def log_metrics_and_artifacts(model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, model_identifier, num_classes, run_name, output, step=None, run_attack=True,
                              attack_pre_trained=False, logistic_regression=False, compute_efficacy=False):
    # store all metrics and artifacts in a dictionary for further use
    metrics_and_artifacts = dict()
    metrics_and_artifacts = log_overall_and_class_accuracies(metrics_and_artifacts, model, x_training, y_training, x_test, y_test, x_remaining, y_remaining, x_target, y_target, model_identifier,
                                                             num_classes, run_name, output, step, logistic_regression)
    if compute_efficacy:
        metrics_and_artifacts = log_efficacy_and_upper_bound(metrics_and_artifacts, model, x_target, y_target, model_identifier, run_name, output, logistic_regression)
    else:
        metrics_and_artifacts['efficacy'] = -1
        metrics_and_artifacts['efficacy_upper_bound'] = -1

    # if the attacks are performed on the pre-trained model, use the original training dataset to train the attack model,
    # otherwise use the remaining dataset since it corresponds to the "training set" of the scrubbed / retrained model
    if run_attack:
        if attack_pre_trained:
            metrics_and_artifacts = log_membership_inference_and_model_inversion(metrics_and_artifacts, model, x_training, y_training, x_test, y_test, x_target, y_target, model_identifier, run_name,
                                                                                 output, step)
        else:
            metrics_and_artifacts = log_membership_inference_and_model_inversion(metrics_and_artifacts, model, x_remaining, y_remaining, x_test, y_test, x_target, y_target, model_identifier, run_name,
                                                                                 output, step)
    else:
        # fill dictionary with dummy values
        metrics_and_artifacts['membership_inference_probabilities'] = 0
        metrics_and_artifacts['membership_inference_probability_mean'] = 0
        metrics_and_artifacts['membership_inference_probability_std'] = 0
        metrics_and_artifacts['extracted_data'] = []
        metrics_and_artifacts['extracted_data_labels'] = []

    return metrics_and_artifacts
