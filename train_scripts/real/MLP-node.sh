device=$1
epsilon=16

# ##################################
# facebook

python train.py mlp-dp \
    --dataset facebook \
    --epsilon $epsilon \
    --num_layers 3 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs 100 \
    --batch_size 256 \
    --device $device \
    --project MLP/node/$epsilon

##################################
# cora

python train.py mlp-dp \
    --dataset cora \
    --epsilon $epsilon \
    --num_layers 3 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs 100 \
    --batch_size 64 \
    --device $device \
    --project MLP/node/$epsilon

##################################
# pubmed

python train.py mlp-dp \
    --dataset pubmed \
    --epsilon $epsilon \
    --num_layers 3 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs 100 \
    --batch_size 256 \
    --device $device \
    --project MLP/node/$epsilon

##################################
# photo

python train.py mlp-dp \
    --dataset photo \
    --epsilon $epsilon \
    --num_layers 3 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs 100 \
    --batch_size 64 \
    --device $device \
    --project MLP/node/$epsilon
    
##################################
# computers

python train.py mlp-dp \
    --dataset computers \
    --epsilon $epsilon \
    --num_layers 3 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs 100 \
    --batch_size 64 \
    --device $device \
    --project MLP/node/$epsilon
    
##################################
# squirrel

python train.py mlp-dp \
    --dataset squirrel \
    --epsilon $epsilon \
    --num_layers 3 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs 100 \
    --batch_size 64 \
    --device $device \
    --project MLP/node/$epsilon
    
##################################
# chameleon

python train.py mlp-dp \
    --dataset chameleon \
    --epsilon $epsilon \
    --num_layers 3 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs 100 \
    --batch_size 64 \
    --device $device \
    --project MLP/node/$epsilon
