import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10


def build_batches(num_data, batch_size):
    """Generate batch indices that can be used to divide the data into batches."""
    permuted_indices = torch.randperm(num_data)
    batch_indices = [permuted_indices[start:start+batch_size] for start in range(0, num_data, batch_size)]
    return batch_indices


def load_data(dataset, num_forget=10, small=True):
    """
    Load specified data set. For MNIST and CIFAR10 transform data into greyscale images and then into one-dimensional vectors.
    """
    dataset = dataset.lower()
    if dataset == 'mnist':
        training_data = MNIST(root='data/', train=True, download=True)
        test_data = MNIST(root='data/', train=False, download=True)
        # reshape MNIST data such that each image is an one-dimensional vector of length 784
        x_training, y_training = torch.reshape(training_data.data, [-1, 784]).float() / 255, training_data.targets.long()
        x_test, y_test = torch.reshape(test_data.data, [-1, 784]).float() / 255, test_data.targets.long()

        if small:
            x_training_small, y_training_small = [], []
            for i in range(10):
                x_training_small.append(x_training[y_training == i][:100])
                y_training_small.append(y_training[y_training == i][:100])
            x_training = torch.concat(x_training_small, dim=0)
            y_training = torch.concat(y_training_small, dim=0)

        return x_training, y_training, x_test, y_test
    elif dataset == 'cifar10':
        # transform images from RGB to greyscale on load
        training_data = CIFAR10(root='data/', train=True, download=True)
        test_data = CIFAR10(root='data/', train=False, download=True)

        # reshape CIFAR10 data such that each image is an one-dimensional vector of length 1024 with values from 0 to 1
        grayscale = torchvision.transforms.Grayscale(num_output_channels=1)
        x_training = torch.tensor(training_data.data).reshape([-1, 3, 32, 32])
        x_training = torch.squeeze(grayscale(x_training))
        x_training = torch.reshape(x_training, [-1, 1024]) / 255
        y_training = torch.tensor(training_data.targets).long()

        x_test = torch.tensor(test_data.data).reshape([-1, 3, 32, 32])
        x_test = torch.squeeze(grayscale(x_test))
        x_test = torch.reshape(x_test, [-1, 1024]) / 255
        y_test = torch.tensor(test_data.targets).long()

        if small:
            x_training_small, y_training_small = [], []
            for i in range(10):
                x_training_small.append(x_training[y_training == i][:100])
                y_training_small.append(y_training[y_training == i][:100])
            x_training = torch.concat(x_training_small, dim=0)
            y_training = torch.concat(y_training_small, dim=0)

        return x_training, y_training, x_test, y_test
    else:
        if dataset == 'synthetic_outlier':
            dataset_path = './data/SYNTHETIC/with_outliers'
            outliers = True
        elif dataset == 'synthetic_cluster':
            dataset_path = './data/SYNTHETIC/without_outliers/dataset0'
            if num_forget < 1 or num_forget > 50:
                raise ValueError(f'Number of target data points must be in [1, 50], but got {num_forget}')
            outliers = False
        else:
            raise ValueError(f'Unknown dataset {dataset}!')

        with open(dataset_path + '/data.pkl', 'rb') as data_file:
            x = torch.load(data_file)
        with open(dataset_path + '/labels.pkl', 'rb') as labels_file:
            y = torch.load(labels_file)

        # scale data to unit sphere
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) * 2 - 1

        # get remaining data points
        x_remaining = torch.concat([x[:40], x[50:]]) if outliers else x[num_forget:]
        y_remaining = torch.concat([y[:40], y[50:]]) if outliers else y[num_forget:]
        # convert labels for remaining data points to one hot vectors
        y_remaining = torch.concat([-y_remaining + 1, y_remaining], dim=1)

        # get target data points
        x_target = x[40:50] if outliers else x[:num_forget]
        y_target = y[40:50] if outliers else y[:num_forget]
        # convert labels for target data points to one hot vectors
        y_target = torch.concat([-y_target + 1, y_target], dim=1)

        # convert labels for whole dataset to one hot vectors
        y = torch.concat([-y + 1, y], dim=1)

        return x, y, x_remaining, y_remaining, x_target, y_target


def split_into_remaining_and_target_data(x, y, target_class, percentage):
    """
    Split data into remaining and target data.
    Target data will belong to the given target class and contain the given percentage of data points of that class.
    If target_class = -1, the target data points will be evenly drawn from all classes.
    """
    if target_class not in range(-1, 10):
        raise ValueError(f'Unknown target class {target_class}')
    if percentage <= 0 or percentage > 1:
        raise ValueError(f'Invalid percentage value {percentage}. Percentage value must be in (0, 1].')

    if target_class == -1:
        # number of target data points per class
        num_target_data_per_class = [0] * 10
        for label in range(10):
            num_target_data_per_class[label] = torch.sum(y == label) * percentage

        # target dataset consists of the target data for each class
        x_target = torch.concat([x[y == label][:num_target_data_per_class[label]] for label in range(10)])
        y_target = torch.concat([y[y == label][:num_target_data_per_class[label]] for label in range(10)])

        # indices of target data points
        target_indices = torch.concat([(y == label).nonzero(as_tuple=True)[0][:num_target_data_per_class[label]] for label in range(10)])

        # remaining data points
        x_remaining = torch.concat([x[y == label][num_target_data_per_class[label]:] for label in range(10)])
        y_remaining = torch.concat([y[y == label][num_target_data_per_class[label]:] for label in range(10)])

        # indices of remaining data points
        remaining_indices = torch.concat([(y == label).nonzero(as_tuple=True)[0][num_target_data_per_class[label]:] for label in range(10)])

        return x_target, y_target, target_indices, x_remaining, y_remaining, remaining_indices

    else:
        # get all data points of the target class
        x_target = x[y == target_class]
        y_target = y[y == target_class]

        # get all data points that do not belong to the target class
        x_remaining = x[y != target_class]
        y_remaining = y[y != target_class]

        # get target stop index from percentage value
        target_stop_index = int(x_target.shape[0] * percentage)

        # remaining data consists of all data points that do not belong to the target class and the remaining data points of the target class
        x_remaining = torch.concat([x_remaining, x_target[target_stop_index:]])
        y_remaining = torch.concat([y_remaining, y_target[target_stop_index:]])

        # indices of remaining data points
        remaining_indices = (y == target_class).nonzero(as_tuple=True)[0][target_stop_index:]

        # target data consists of the given percentage of those data points belonging to the target class
        x_target = x_target[:target_stop_index]
        y_target = y_target[:target_stop_index]

        # indices of target data points
        target_indices = (y == target_class).nonzero(as_tuple=True)[0][:target_stop_index]

        return x_target, y_target, target_indices, x_remaining, y_remaining, remaining_indices

