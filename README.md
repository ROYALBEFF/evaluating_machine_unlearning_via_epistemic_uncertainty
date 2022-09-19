# Experiment instructions
This is a step-by-step guide for reproducing the experiments in the paper
"Evaluating Machine Unlearning via Epistemic Uncertainty".
All experiment configurations are pre-defined in the provided shell scripts.

## Installation
1. Install `python` 3.9 and the latest versions of `conda` and `pip` on your machine.
2. Install `mlflow` using `pip`:
```
pip install mlflow==1.23.0
```
3. All requirements will be automatically installed on the first call of one of the scripts below. 
Therefore, the first run might take a little bit extra time.

## Prepare pre-trained models for experiments
To reproduce the experiments in the paper run:
```
./mnist_training.sh
./cifar_training.sh
```

Additional model trainings can be run via:
```
mlflow run . -e training -P seed={SEED} -P dataset={DATASET} -P learning_rate={LR} -P epochs={EPOCHS} -P batch_size={BS} -P output={OUT};
```
You can find the usage help by running:
```
python run_training.py -h
```

**Note: Each batch update will be stored for Amnesiac Unlearning. Depending on the model and the dataset this might lead to a large memory usage!**

## Compute efficacy of initial models
To reproduce the experiments in the paper run:
```
./mnist_initial.sh
./cifar_initial.sh
```

You can compute the efficacy for additional initializations by running:
```
mlflow run . -e initial_efficacy -P seed={SEED} -P dataset={DATASET} -P target={TARGET_CLASS} -P percentage={P} -P output={OUT}
```

You can find the usage help by running:
```
python run_initial_efficacy.py -h
```
If you want to check the initial efficacy for a specific pre-trained model, make sure to provide the same random seed!

## Perform membership inference attack on pre-trained models
To reproduce the experiments in the paper run:
```
./mnist_attack.sh
./cifar_attack.sh
```

You can perform additional membership inference attacks on the pre-trained models by running:
```
mlflow run . -e attack_pre_trained -P seed={SEED} -P model={MODEL} -P dataset={DATASET} -P target={TARGET_CLASS} -P percentage={P} -P output={OUT}
```

You can find the usage help by running:
```
python run_attack_pre_trained.py -h
```

## Perform forgetting algorithms
You can find the usage help for all forgetting algorithms by running:
```
python run_forgetting.py -h
```

### Retraining
To reproduce the experiments in the paper run:
```
./mnist_retraining.sh
./cifar_retraining.sh
```

You can perform additional retrainings by running:
```
mlflow run . -e forgetting -P seed={SEED} -P model={MODEL} -P dataset={DATASET} -P target={TARGET_CLASS} -P percentage={P} -P forgetting=retraining -P learning_rate={LR} -P epochs={EPOCHS} -P batch_size={BS} -P output={OUT}
```
Make sure to use the same hyper-parameters and random seed as for the original training!


### Fisher Forgetting
To reproduce the experiments in the paper run:
```
./mnist_fisher.sh
./cifar_fisher.sh
```

You can perform additional Fisher Forgetting experiments by running:
```
mlflow run . -e forgetting -P seed={SEED} -P model={MODEL} -P dataset={DATASET} -P target={TARGET_CLASS} -P percentage={P} -P forgetting=fisher -P alpha={A} -P output={OUT}
```

### Amnesiac Unlearning
To reproduce the experiments in the paper run:
```
./mnist_amnesiac.sh
./cifar_amnesiac.sh
```

You can perform additional Amnesiac Unlearning experiments by running:
```
mlflow run . -e forgetting -P seed={SEED} -P model={MODEL} -P dataset={DATASET} -P target={TARGET_CLASS} -P percentage={P} -P forgetting=amnesiac -P history={BATCH_UPDATES} -P output={OUT}
```

## Create plots
To plot the experimental results as seen in the paper run:
```
python create_plots.py -r {PATH_TO_EXPERIMENTS} -o {OUT}
```
**Note that you won't obtain the exact same plots as in the paper, since we adjusted the axes' limits for nicer presentation.**

## Inspect experiments using mlfow
Finally, if you want to examine any of your experiments in detail, use the mlflow dashboard. Run `mlflow ui`
in the directory containing `mlruns`. This should be the project directory by default. Afterwards, open `127.0.0.1:5000` in your browser.