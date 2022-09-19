import argparse
import pickle
import os
from collections import defaultdict
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pathlib


def unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_experiment_directories(root):
    training_directories = []
    attack_directories = []
    retraining_directories = []
    fisher_directories = []
    amnesiac_directories = []
    init_directories = []

    root = root if not root.endswith('/') else root[:-1]
    for d in os.walk(root):
        if d[0].startswith(f'{root}/training'):
            training_directories.append(d[0])
        elif d[0].startswith(f'{root}/attack'):
            attack_directories.append(d[0])
        elif d[0].startswith(f'{root}/init'):
            init_directories.append(d[0])
        elif 'retraining' in d[0]:
            retraining_directories.append(d[0])
        elif 'fisher' in d[0]:
            fisher_directories.append(d[0])
        elif 'amnesiac' in d[0]:
            amnesiac_directories.append(d[0])

    return training_directories, init_directories, attack_directories, retraining_directories, fisher_directories, amnesiac_directories


def get_efficacies_and_upper_bounds(init_dirs, retraining_dirs, fisher_dirs, amnesiac_dirs):
    init_efficacies = defaultdict(list)
    init_upper_bounds = defaultdict(list)
    pre_trained_efficacies = defaultdict(list)
    pre_trained_upper_bounds = defaultdict(list)
    retraining_efficacies = defaultdict(list)
    retraining_upper_bounds = defaultdict(list)
    amnesiac_efficacies = defaultdict(list)
    amnesiac_upper_bounds = defaultdict(list)
    fisher_efficacies = defaultdict(list)
    fisher_upper_bounds = defaultdict(list)

    for dir in init_dirs:
        name = pathlib.PurePath(dir).name
        percentage = float(name.split('_')[-1])
        init_efficacies[percentage].append(unpickle(f'{dir}/efficacy_init.pkl'))
        init_upper_bounds[percentage].append(unpickle(f'{dir}/efficacy_upper_bound_init.pkl'))

    for dir in retraining_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[7]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        pre_trained_efficacies[percentage].append(unpickle(f'{dir}/efficacy_pre-trained.pkl'))
        pre_trained_upper_bounds[percentage].append(unpickle(f'{dir}/efficacy_upper_bound_pre-trained.pkl'))
        retraining_efficacies[percentage].append(unpickle(f'{dir}/efficacy_progression_retraining.pkl')[-1])
        retraining_upper_bounds[percentage].append(unpickle(f'{dir}/efficacy_upper_bound_progression_retraining.pkl')[-1])

    for dir in fisher_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[7]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        fisher_efficacies[percentage].append(unpickle(f'{dir}/efficacy_fisher.pkl'))
        fisher_upper_bounds[percentage].append(unpickle(f'{dir}/efficacy_upper_bound_fisher.pkl'))

    for dir in amnesiac_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[7]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        amnesiac_efficacies[percentage].append(unpickle(f'{dir}/efficacy_amnesiac.pkl'))
        amnesiac_upper_bounds[percentage].append(unpickle(f'{dir}/efficacy_upper_bound_amnesiac.pkl'))

    return init_efficacies, init_upper_bounds, pre_trained_efficacies, pre_trained_upper_bounds, retraining_efficacies, retraining_upper_bounds,\
           amnesiac_efficacies, amnesiac_upper_bounds, fisher_efficacies, fisher_upper_bounds


def get_accuracies(retraining_dirs, amnesiac_dirs, fisher_dirs):
    pre_trained_overall_remaining_accuracy = defaultdict(list)
    pre_trained_overall_target_accuracy = defaultdict(list)
    pre_trained_overall_test_accuracy = defaultdict(list)
    retraining_overall_remaining_accuracy = defaultdict(list)
    retraining_overall_target_accuracy = defaultdict(list)
    retraining_overall_test_accuracy = defaultdict(list)
    amnesiac_overall_remaining_accuracy = defaultdict(list)
    amnesiac_overall_target_accuracy = defaultdict(list)
    amnesiac_overall_test_accuracy = defaultdict(list)
    fisher_overall_remaining_accuracy = defaultdict(list)
    fisher_overall_target_accuracy = defaultdict(list)
    fisher_overall_test_accuracy = defaultdict(list)

    for dir in retraining_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[7]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        pre_trained_overall_remaining_accuracy[percentage].append(unpickle(f'{dir}/remaining_accuracy_pre-trained.pkl')[0])
        pre_trained_overall_target_accuracy[percentage].append(unpickle(f'{dir}/target_accuracy_pre-trained.pkl')[0])
        pre_trained_overall_test_accuracy[percentage].append(unpickle(f'{dir}/test_accuracy_pre-trained.pkl')[0])
        retraining_overall_remaining_accuracy[percentage].append(unpickle(f'{dir}/overall_remaining_accuracy_progression_retraining.pkl')[-1])
        retraining_overall_target_accuracy[percentage].append(unpickle(f'{dir}/overall_target_accuracy_progression_retraining.pkl')[-1])
        retraining_overall_test_accuracy[percentage].append(unpickle(f'{dir}/overall_test_accuracy_progression_retraining.pkl')[-1])

    for dir in amnesiac_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[7]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        amnesiac_overall_target_accuracy[percentage].append(unpickle(f'{dir}/target_accuracy_amnesiac.pkl')[0])
        amnesiac_overall_remaining_accuracy[percentage].append(unpickle(f'{dir}/remaining_accuracy_amnesiac.pkl')[0])
        amnesiac_overall_test_accuracy[percentage].append(unpickle(f'{dir}/test_accuracy_amnesiac.pkl')[0])

    for dir in fisher_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[7]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        fisher_overall_target_accuracy[percentage].append(unpickle(f'{dir}/target_accuracy_fisher.pkl')[0])
        fisher_overall_remaining_accuracy[percentage].append(unpickle(f'{dir}/remaining_accuracy_fisher.pkl')[0])
        fisher_overall_test_accuracy[percentage].append(unpickle(f'{dir}/test_accuracy_fisher.pkl')[0])

    return pre_trained_overall_remaining_accuracy, pre_trained_overall_target_accuracy, pre_trained_overall_test_accuracy, \
           retraining_overall_remaining_accuracy, retraining_overall_target_accuracy, retraining_overall_test_accuracy, \
           amnesiac_overall_remaining_accuracy, amnesiac_overall_target_accuracy, amnesiac_overall_test_accuracy, \
           fisher_overall_remaining_accuracy, fisher_overall_target_accuracy, fisher_overall_test_accuracy


def get_membership_inference_means(attack_dirs, retraining_dirs, amnesiac_dirs, fisher_dirs):
    pre_trained_mia_means = defaultdict(list)
    retraining_mia_means = defaultdict(list)
    fisher_mia_means = defaultdict(list)
    amnesiac_mia_means = defaultdict(list)

    for dir in retraining_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[7]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        retraining_mia_means[percentage].append(unpickle(f'{dir}/membership_inference_probability_mean_progression_retraining.pkl')[-1])

    for dir in fisher_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[7]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        fisher_mia_means[percentage].append(np.mean(unpickle(f'{dir}/membership_inference_probabilities_fisher.pkl')))

    for dir in amnesiac_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[7]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        amnesiac_mia_means[percentage].append(np.mean(unpickle(f'{dir}/membership_inference_probabilities_amnesiac.pkl')))

    for dir in attack_dirs:
        name = pathlib.PurePath(dir).name
        percentage = name.split('_')[8]
        percentage = 1 if percentage.startswith('1') else float(percentage[0] + '.' + percentage[1:])
        pre_trained_mia_means[percentage].append(np.mean(unpickle(f'{dir}/membership_inference_probabilities_pre-trained.pkl')))

    return pre_trained_mia_means, retraining_mia_means, fisher_mia_means, amnesiac_mia_means


def plot_mia_mean(prob_means, title, out, x_lim=None):
    plt.clf()
    for p in [1, 0.8, 0.5, 0.25, 0.1, 0.01]:
        ax = sns.distplot(prob_means[p], hist=False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.set_xlabel('MIA mean probability')
    ax.legend(['1.0', '0.8', '0.5', '0.25', '0.1', '0.01'])
    ax.set_title(title)
    plt.savefig(f'{args.o}/{out}')


def plot_efficacy(efficacies, title, out, x_lim=None):
    plt.clf()
    # percentages = [1, 0.8, 0.5, 0.25, 0.1, 0.01]
    percentages = [1, 0.8, 0.5, 0.25, 0.1]
    for p in percentages:
        ax = sns.distplot(x=efficacies[p], hist=False)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.legend(['1.0', '0.8', '0.5', '0.25', '0.1', '0.01'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Efficacy')
    plt.savefig(f'{args.o}/{out}')
    ax.set_title(title)


def plot_efficacy_and_upper_bound(efficacies, upper_bounds, title, out, x_lim=None):
    plt.clf()
    with sns.color_palette("Paired"):
        # percentages = [1, 0.8, 0.5, 0.25, 0.1, 0.01]
        percentages = [1, 0.8, 0.5, 0.25, 0.1]
        for p in percentages:
            ax = sns.distplot(x=upper_bounds[p], kde_kws={'linestyle': '--'}, hist=False)
            ax = sns.distplot(x=efficacies[p], hist=False)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.legend(['1.0', '1.0', '0.8', '0.8', '0.5', '0.5', '0.25', '0.25', '0.1', '0.1'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Efficacy')
        ax.set_title(title)
        plt.savefig(f'{args.o}/{out}')


def plot_efficacy_comparison(init_efficacies, pre_trained_efficacies, retraining_efficacies, amnesaic_efficacies, fisher_efficacies):
    plt.clf()
    sns.distplot(x=init_efficacies[1], hist=False)
    sns.distplot(x=pre_trained_efficacies[1], hist=False)
    sns.distplot(x=retraining_efficacies[1], hist=False)
    sns.distplot(x=amnesaic_efficacies[1], hist=False)
    ax = sns.distplot(x=fisher_efficacies[1], hist=False)
    ax.legend(['Initial', 'Pre-trained', 'Retraining', 'Amnesiac Unlearning', 'Fisher Forgetting'])
    ax.set_xlabel('Efficacy')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Efficacy comparison')
    plt.savefig(f'{args.o}/efficacy_comparison.png')


def plot_efficacy_mia(efficacies, prob_means, title, out):
    plt.clf()
    for p in [1, 0.8, 0.5, 0.25, 0.1, 0.01]:
        ax = sns.scatterplot(x=efficacies[p], y=prob_means[p])
    ax.legend(['1.0', '0.8', '0.5', '0.25', '0.1', '0.01'])
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Efficacy')
    ax.set_ylabel('MIA mean probability')
    plt.savefig(f'{args.o}/{out}')


def main():
    # create output path
    pathlib.Path(args.o).mkdir(parents=True, exist_ok=True)

    training_dirs, init_dirs, attack_dirs, retraining_dirs, fisher_dirs, amnesiac_dirs = get_experiment_directories(args.r)

    # plot efficacies, upper bounds and comparison
    init_efficacies, init_upper_bounds, pre_trained_efficacies, pre_trained_upper_bounds, retraining_efficacies, retraining_upper_bounds, \
    amnesiac_efficacies, amnesiac_upper_bounds, fisher_efficacies, fisher_upper_bounds = get_efficacies_and_upper_bounds(init_dirs,retraining_dirs, fisher_dirs, amnesiac_dirs)

    plot_efficacy(pre_trained_efficacies, 'Efficacy - Pre-trained', 'efficacy_pre_trained.png')
    plot_efficacy(retraining_efficacies, 'Efficacy - Retraining', 'efficacy_retraining.png')
    plot_efficacy(amnesiac_efficacies, 'Efficacy - Amnesiac', 'efficacy_amnesiac.png')
    plot_efficacy(fisher_efficacies, 'Efficacy - Fisher', 'efficacy_fisher.png')

    plot_efficacy_and_upper_bound(pre_trained_efficacies, pre_trained_upper_bounds, 'Efficacy and upper bound - Pre-trained', 'efficacy_ub_pre_trained.png')
    plot_efficacy_and_upper_bound(retraining_efficacies, retraining_upper_bounds, 'Efficacy and upper bound - Retraining', 'efficacy_ub_retraining.png')
    plot_efficacy_and_upper_bound(amnesiac_efficacies, amnesiac_upper_bounds, 'Efficacy and upper bound - Amnesiac', 'efficacy_ub_amnesiac.png')
    plot_efficacy_and_upper_bound(fisher_efficacies, fisher_upper_bounds, 'Efficacy and upper bound - Fisher', 'efficacy_ub_fisher.png')

    plot_efficacy_comparison(init_efficacies, pre_trained_efficacies, retraining_efficacies, amnesiac_efficacies, fisher_efficacies)

    # plot membership inference mean probabilities
    pre_trained_mia_means, retraining_mia_means, fisher_mia_means, amnesiac_mia_means = get_membership_inference_means(attack_dirs, retraining_dirs, amnesiac_dirs, fisher_dirs)

    plot_mia_mean(pre_trained_mia_means, 'MIA mean probability - Pre-trained', 'mia_pre_trained.png')
    plot_mia_mean(retraining_mia_means, 'MIA mean probability - Retraining', 'mia_retrained.png')
    plot_mia_mean(amnesiac_mia_means, 'MIA mean probability - Amnesiac', 'mia_amnesiac.png')
    plot_mia_mean(fisher_mia_means, 'MIA mean probability - Fisher', 'mia_fisher.png')

    # plot relation of mean probabilities and efficacies
    plot_efficacy_mia(pre_trained_efficacies, pre_trained_mia_means, 'Efficacy X MIA mean probability - Pre-trained', 'efficacy_mia_pre_trained.png')
    plot_efficacy_mia(retraining_efficacies, retraining_mia_means, 'Efficacy X MIA mean probability - Retraining', 'efficacy_mia_retraining.png')
    plot_efficacy_mia(amnesiac_efficacies, amnesiac_mia_means, 'Efficacy X MIA mean probability - Amnesiac', 'efficacy_mia_amnesiac.png')
    plot_efficacy_mia(fisher_efficacies, fisher_mia_means, 'Efficacy X MIA mean probability - Fisher', 'efficacy_mia_fisher.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create plots as seen in the paper.')
    parser.add_argument('-r', '-root', type=str, required=True, help='Root directory containing all experiment directories.')
    parser.add_argument('-o', '-output', type=str, required=False, default='.', help='Output directory.')
    args = parser.parse_args()

    plt.rcParams['figure.dpi'] = 300
    main()
