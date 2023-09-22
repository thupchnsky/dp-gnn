nk=$1
device=$2
phi=$3
epsilon=16
epoch=100

# cSBM

CUDA_VISIBLE_DEVICES=$device python train.py dpdgc-edp \
    --dataset cSBM \
    --hidden-dim 64 \
    --dropout 0.5 \
    --encoder-epochs $epoch \
    --encoder-lr 1e-3 \
    --epochs $epoch \
    --learning-rate 1e-3 \
    --repeats 10 \
    --epsilon $epsilon \
    --batch-size 256 \
    --device cuda \
    --norm-scale 1e-8 \
    --project DPDGC/edge/csbm/$phi/$epsilon/ \
    --csbm_phi $phi



