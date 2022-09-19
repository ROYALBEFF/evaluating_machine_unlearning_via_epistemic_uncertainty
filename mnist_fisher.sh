declare -a seeds=("250120" "250121" "250122" "250123" "250124" "250125" "250126" "250127" "250128" "250129" "250130" "250131" "250132" "250133" "250134" "250135" "250136" "250137" "250138" "250139")
declare -a percentages=("0.01" "0.1" "0.25" "0.5" "0.8" "1")

for s in "${seeds[@]}"
do
  for p in "${percentages[@]}"
  do
    model="training_s_${s}_d_mnist_lr_0_1_e_50_bs_32"
    mlflow run . -e forgetting -P seed="$s" -P model="./mnist/models/${model}/pre-trained.pth" -P dataset=mnist -P target=3 -P percentage="$p" -P forgetting=fisher -P alpha=3e-6 -P output=./mnist/
  done
done
