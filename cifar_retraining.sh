declare -a seeds=("250120" "250121" "250122" "250123" "250124" "250125" "250126" "250127" "250128" "250129" "250130" "250131" "250132" "250133" "250134" "250135" "250136" "250137" "250138" "250139")
declare -a percentages=("0.01" "0.1" "0.25" "0.5" "0.8" "1")

for s in "${seeds[@]}"
do
  for p in "${percentages[@]}"
  do
    # retrain CIFAR10 model
    model="training_s_${s}_d_cifar10_lr_0_1_e_200_bs_64/pre-trained.pth"
    mlflow run . -e forgetting -P seed="$s" -P model="./cifar10/models/${model}" -P dataset=cifar10 -P target=8 -P percentage="$p" -P forgetting=retraining -P learning_rate=0.1 -P epochs=200 -P batch_size=64 -P output=./cifar10/
  done
done
