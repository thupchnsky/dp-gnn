nk=$1
device=$2
phi=$3
epsilon=16
epoch=100

# cSBM

CUDA_VISIBLE_DEVICES=$device python train.py dpdgc-ndp \
    --dataset cSBM \
    --epsilon $epsilon \
    --hidden-dim 64 \
    --dropout 0.0 \
    --encoder-epochs $epoch \
    --encoder-lr 1e-3 \
    --epochs $epoch \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size 256 \
    --device cuda \
    --norm-scale 1e-8 \
    --max_degree 100 \
    --project DPDGC/node/csbm/$phi/$epsilon/ \
    --csbm_phi $phi
