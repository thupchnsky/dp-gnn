nk=$1
device=$2
phi=$3
epsilon=16
epoch=100
degree=100
hops=2

# cSBM

CUDA_VISIBLE_DEVICES=$device python train.py gap-ndp \
    --dataset cSBM \
    --epsilon $epsilon \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --max_degree $degree \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --max_grad_norm 1 \
    --epochs $epoch \
    --batch_size 256 \
    --encoder_epochs $epoch \
    --device cuda \
    --project GAP/node/csbm/$phi/$hops/$epsilon/ \
    --csbm_phi $phi
