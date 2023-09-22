device=$1
epsilon=16
epoch=100
degree=100

###################################################
# facebook
python train.py sage-ndp \
    --dataset facebook \
    --epsilon $epsilon \
    --base_layers 2 \
    --head_layers 1 \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 256 \
    --device $device \
    --project SAGE/node/$epsilon

###################################################
# cora
python train.py sage-ndp \
    --dataset cora \
    --epsilon $epsilon \
    --base_layers 2 \
    --head_layers 1 \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --device $device \
    --project SAGE/node/$epsilon

###################################################
# pubmed
python train.py sage-ndp \
    --dataset pubmed \
    --epsilon $epsilon \
    --base_layers 2 \
    --head_layers 1 \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 256 \
    --device $device \
    --project SAGE/node/$epsilon

###################################################
# photo
python train.py sage-ndp \
    --dataset photo \
    --epsilon $epsilon \
    --base_layers 2 \
    --head_layers 1 \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --device $device \
    --project SAGE/node/$epsilon

###################################################
# computers
python train.py sage-ndp \
    --dataset computers \
    --epsilon $epsilon \
    --base_layers 2 \
    --head_layers 1 \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --device $device \
    --project SAGE/node/$epsilon

###################################################
# squirrel
python train.py sage-ndp \
    --dataset squirrel \
    --epsilon $epsilon \
    --base_layers 2 \
    --head_layers 1 \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --device $device \
    --project SAGE/node/$epsilon
    
###################################################
# chameleon
python train.py sage-ndp \
    --dataset chameleon \
    --epsilon $epsilon \
    --base_layers 2 \
    --head_layers 1 \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --device $device \
    --project SAGE/node/$epsilon