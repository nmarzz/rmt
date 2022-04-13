# Run experiment trying all dropouts

for dataset in mnist fashion_mnist cifar10
do
  for dropout in no_dropout pytorch k_bernoulli rbf sine
  do
    for i in {0..4}    
    do    
      python3 run_training.py --model 'mlp' --dataset $dataset --batch-size 256 --device 'cuda:0' --dropout-proportion 0.5 --dropout-type $dropout  --epochs 50
    done  
  done
done
