import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import pathlib
import torch
from model import LogisticRegression


def get_plot_prefix(run_name, output):
    plot_prefix = f'{output}/plots/{run_name}/'
    # create directory if necessary
    pathlib.Path(plot_prefix).mkdir(parents=True, exist_ok=True)
    return plot_prefix


def plot_accuracy_progression(metric_progressions, steps, num_classes, model_identifier, run_name, output):
    # clear figure and plot accuracy progressions
    plt.clf()
    step_range = list(range(steps))
    plot_prefix = get_plot_prefix(run_name, output)

    # plot overall accuracies for all datasets
    sns.lineplot(x=step_range, y=metric_progressions.overall_test_accuracy_progression)
    sns.lineplot(x=step_range, y=metric_progressions.overall_training_accuracy_progression)
    sns.lineplot(x=step_range, y=metric_progressions.overall_remaining_accuracy_progression)
    ax = sns.lineplot(x=step_range, y=metric_progressions.overall_target_accuracy_progression)
    ax.set_title('Overall accuracy progression')
    ax.legend(labels=['Test data', 'Training data', 'Remaining data', 'Target data'])

    # save and log plot
    plot_name = f'{model_identifier}_overall_accuracy_progression.png'
    plot_path = plot_prefix + plot_name
    mlflow.log_figure(ax.figure, f'plots/{plot_name}')
    plt.savefig(plot_path)

    # plot class accuracies for all datasets
    for label in range(num_classes):
        plt.clf()
        sns.lineplot(x=step_range, y=metric_progressions.class_test_accuracy_progressions[label])
        sns.lineplot(x=step_range, y=metric_progressions.class_training_accuracy_progressions[label])
        sns.lineplot(x=step_range, y=metric_progressions.class_remaining_accuracy_progressions[label])
        ax = sns.lineplot(x=step_range, y=metric_progressions.class_target_accuracy_progressions[label])
        ax.set_title(f'Accuracies for class {label}')
        ax.legend(labels=['Test data', 'Training data', 'Remaining data', 'Target data'])

        # save and log plot
        plot_name = f'{model_identifier}_accuracy_progression_class_{label}.png'
        plot_path = plot_prefix + plot_name
        mlflow.log_figure(ax.figure, f'plots/{plot_name}')
        plt.savefig(plot_path)


def plot_membership_inference_probabilities(membership_inference_probabilities, model_identifier, run_name, output, step=None):
    plt.clf()
    ax = sns.histplot(x=membership_inference_probabilities, kde=True)
    ax.set_title('Membership inference probabilities')

    # save and log plot
    plot_name = f'{model_identifier}_membership_inference_probabilities'
    plot_name += '' if step is None else f'_{step}.png'
    plot_name += '.png'

    plot_path = get_plot_prefix(run_name, output) + plot_name
    mlflow.log_figure(ax.figure, f'plots/{plot_name}')
    plt.savefig(plot_path)


def plot_membership_inference_probability_mean_and_std_progression(metric_progressions, steps, model_identifier, run_name, output):
    # clear figure and membership inference probability mean and standard deviation progression
    plt.clf()
    step_range = list(range(steps))
    sns.lineplot(x=step_range, y=metric_progressions.membership_inference_probability_mean_progression)
    ax = sns.lineplot(x=step_range, y=metric_progressions.membership_inference_probability_std_progression)
    ax.set_title('Membership inference probability mean and standard deviation progression')
    ax.legend(labels=['Mean', 'Standard deviation'])

    # save and log plot
    plot_name = f'{model_identifier}_membership_inference_probability_mean_and_std_progression.png'
    plot_path = get_plot_prefix(run_name, output) + plot_name
    mlflow.log_figure(ax.figure, f'plots/{plot_name}')
    plt.savefig(plot_path)


def plot_model_inversion_extracted_data(x_extracted, y_extracted, shape, model_identifier, run_name, output, step=None):
    plot_prefix = get_plot_prefix(run_name, output)
    for i, x in enumerate(x_extracted):
        # clear figure
        plt.clf()

        # reshape data point from one to two dimensions
        x = np.reshape(x, shape)
        y = y_extracted[i]

        # save and log plot
        ax = plt.imshow(x, cmap='gray')
        ax.figure.suptitle(f'Extracted image of class {y}')
        plot_name = f'extracted_data_class_{y}_{model_identifier}'
        plot_name += '' if step is None else f'_step_{step}'
        plot_name += f'{i}.png'
        plot_path = plot_prefix + plot_name
        mlflow.log_figure(ax.figure, f'plots/{plot_name}')
        plt.savefig(plot_path)


def plot_gradient_norm_progression(metric_progressions, run_name, output):
    # clear figure and plot gradient norm progression
    plt.clf()
    range_steps = list(range(len(metric_progressions.gradient_norm_progression)))
    ax = sns.lineplot(x=range_steps, y=metric_progressions.gradient_norm_progression)
    ax.set_title('Gradient norm progression')
    ax.legend(labels=['Gradient norm'])

    # save and log plot
    plot_prefix = get_plot_prefix(run_name, output)
    plot_name = 'gradient_norm_progression.png'
    plot_path = plot_prefix + plot_name
    mlflow.log_figure(ax.figure, f'plots/{plot_name}')
    plt.savefig(plot_path)


def plot_efficacy_progression(metric_progressions, model_identifier, run_name, output):
    # clear figure and plot gradient norm progression
    plt.clf()
    range_steps = list(range(len(metric_progressions.efficacy_progression)))
    # plot efficacy progression
    sns.lineplot(x=range_steps, y=metric_progressions.efficacy_progression)
    # plot efficacy upper bound progression
    ax = sns.lineplot(x=range_steps, y=metric_progressions.efficacy_progression)
    ax.set_title('Efficacy and upper bound progression')
    ax.legend(labels=['Efficacy', 'Efficacy upper bound'])

    # save and log plot
    plot_name = f'{model_identifier}_efficacy_progression.png'
    plot_path = get_plot_prefix(run_name, output) + plot_name
    mlflow.log_figure(ax.figure, f'plots/{plot_name}')
    plt.savefig(plot_path)


def plot_decision_boundary(model, x, y, model_identifier, run_name, output):
    plt.clf()

    # create grid from -1 to 1
    x_grid = torch.linspace(-1, 1, 100)
    x_grid, y_grid = torch.meshgrid(x_grid, x_grid)
    x_grid, y_grid = torch.vstack([x_grid.ravel(), y_grid.ravel()])
    grid = torch.tensor(list(zip(x_grid, y_grid)))

    if isinstance(model, LogisticRegression):
        # compute and plot decision boundary of logistic regression model
        decision_boundary = model.get_decision_boundary()
        sns.lineplot(x=torch.linspace(-1, 1, 10), y=decision_boundary)
    else:
        # get predicted class labels for all points on the grid and plot grid
        grid_predictions = torch.argmax(torch.softmax(model(grid), dim=1), dim=1)
        sns.scatterplot(x=grid[:, 0], y=grid[:, 1], hue=grid_predictions.flatten())

    # one-hot vector to class label
    y = torch.argmax(y, dim=1)
    ax = sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y.flatten())
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)

    # save and log plot
    plot_name = f'decision_boundary_{model_identifier}.png'
    plot_path = get_plot_prefix(run_name, output) + plot_name
    mlflow.log_figure(ax.figure, f'plots/{plot_name}')
    plt.savefig(plot_path)


def plot_logistic_regression_model_comparison(x, y, x_target, original_model_decision_boundary, retrained_model_decision_boundary, scrubbed_model_decision_boundary, run_name, output):
    plt.clf()

    # plot all three decision boundaries
    sns.lineplot(x=np.linspace(-1, 1, 10), y=original_model_decision_boundary, color='grey')
    sns.lineplot(x=np.linspace(-1, 1, 10), y=retrained_model_decision_boundary, color='green')
    ax = sns.lineplot(x=np.linspace(-1, 1, 10), y=scrubbed_model_decision_boundary)
    ax.legend(labels=['Original', 'Retraining', 'Forgetting'])

    # plot data points
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y.flatten())

    # highlight target data points
    sns.scatterplot(x=x_target[:, 0], y=x_target[:, 1], color='red')
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)

    # save and log plot
    plot_name = 'logistic_regression_model_comparison.png'
    plot_path = get_plot_prefix(run_name, output) + plot_name
    mlflow.log_figure(ax.figure, f'plots/{plot_name}')
    plt.savefig(plot_path)
