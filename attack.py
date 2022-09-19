import torch.nn
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.model_inversion import MIFace
from art.estimators.classification import PyTorchClassifier
import numpy as np


def onehot(y, num_classes):
    labels = np.zeros([y.shape[0], num_classes])
    for i, label in enumerate(y):
        labels[i][label] = 1
    return labels


def membership_inference_attack(model, x_training, y_training, x_test, y_test, x_attack, y_attack):
    """
    Train attack model (neural network) and perform membership inference attack on target model.
    x_training, y_training will be used to train the attack model. Corresponds to the training set of the target model.
    x_test, y_test will be used to evaluate the attack model. Dataset that wasn't used for training the target model.
    x_attack, y_attack will be used for the actual attack.
    """
    num_classes = 10
    input_dim = x_training.shape[-1]
    target = PyTorchClassifier(model=model, loss=torch.nn.CrossEntropyLoss(), input_shape=(input_dim,), nb_classes=num_classes)
    attack = MembershipInferenceBlackBox(target, attack_model_type='nn', input_type='prediction')

    y_training = onehot(y_training, num_classes)
    y_test = onehot(y_test, num_classes)
    y_attack = onehot(y_attack, num_classes)

    attack.fit(x=x_training, y=y_training, test_x=x_test, test_y=y_test)
    return attack.infer(x_attack, y_attack, probabilities=True), attack


def model_inversion_attack(model, y_attack, input_dim):
    """
    Train attack model (neural network) and perform model inversion attack on target model.
    y_attack are the labels for which we want the attack model to extract data points from the target model.
    """
    num_classes = 10
    target = PyTorchClassifier(model=model, loss=torch.nn.CrossEntropyLoss(), input_shape=(input_dim,), nb_classes=num_classes)
    attack = MIFace(target)
    y_attack = onehot(y_attack, num_classes)[:1]
    return attack.infer(x=None, y=y_attack), attack
