device=$1
epsilon=16
epoch=100
degree=100
hops=2


###################################################
# facebook
python train.py gap-ndp \
    --dataset facebook \
    --epsilon $epsilon \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 256 \
    --encoder_epochs $epoch \
    --device $device \
    --project GAP/node/$epsilon/$hops

###################################################
# cora
python train.py gap-ndp \
    --dataset cora \
    --epsilon $epsilon \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --encoder_epochs $epoch \
    --device $device \
    --project GAP/node/$epsilon/$hops

###################################################
# pubmed
python train.py gap-ndp \
    --dataset pubmed \
    --epsilon $epsilon \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 256 \
    --encoder_epochs $epoch \
    --device $device \
    --project GAP/node/$epsilon/$hops

###################################################
# photo
python train.py gap-ndp \
    --dataset photo \
    --epsilon $epsilon \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --encoder_epochs $epoch \
    --device $device \
    --project GAP/node/$epsilon/$hops

###################################################
# computers
python train.py gap-ndp \
    --dataset computers \
    --epsilon $epsilon \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --encoder_epochs $epoch \
    --device $device \
    --project GAP/node/$epsilon/$hops

###################################################
# squirrel
python train.py gap-ndp \
    --dataset squirrel \
    --epsilon $epsilon \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --encoder_epochs $epoch \
    --device $device \
    --project GAP/node/$epsilon/$hops

###################################################
# chameleon
python train.py gap-ndp \
    --dataset chameleon \
    --epsilon $epsilon \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --max_degree 100 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 64 \
    --encoder_epochs $epoch \
    --device $device \
    --project GAP/node/$epsilon/$hops
