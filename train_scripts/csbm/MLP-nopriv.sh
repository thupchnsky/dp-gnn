nk=$1
device=$2
phi=$3
epsilon=16
epoch=100

# cSBM

OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=$device python train.py mlp \
    --dataset cSBM \
    --num_layers 3 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --epochs 100 \
    --batch_size full \
    --dropout 0.5 \
    --device cuda \
    --project MLP/nopriv/csbm/$phi \
    --csbm_phi $phi

