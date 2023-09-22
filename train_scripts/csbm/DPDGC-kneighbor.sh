nk=$1
device=$2
phi=$3
epsilon=16
epoch=100

# cSBM

CUDA_VISIBLE_DEVICES=$device python train.py dpdgc-kndp \
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
    --nk $nk \
    --project DPDGC/kneighbor/csbm/$phi/$nk/$epsilon/ \
    --csbm_phi $phi
