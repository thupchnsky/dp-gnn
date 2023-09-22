nk=$1
device=$2
phi=$3
epsilon=16
epoch=100

# cSBM
if [ $phi -eq 0 ]
then
   CUDA_VISIBLE_DEVICES=$device python train.py dpdgc-inf \
    --dataset cSBM \
    --hidden-dim 64 \
    --dropout 0.5 \
    --encoder-epochs 100 \
    --encoder-lr 1e-5 \
    --encoder_dropout 0.5 \
    --epochs 100 \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size full \
    --device cuda \
    --norm-scale 1e-8 \
    --project DPDGC/nopriv/csbm/$phi/ \
    --csbm_phi $phi
else
   CUDA_VISIBLE_DEVICES=$device python train.py dpdgc-inf \
    --dataset cSBM \
    --hidden-dim 64 \
    --dropout 0.5 \
    --encoder-epochs 100 \
    --encoder-lr 1e-3 \
    --encoder_dropout 0.5 \
    --epochs 100 \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size full \
    --device cuda \
    --norm-scale 1e-8 \
    --project DPDGC/nopriv/csbm/$phi/ \
    --csbm_phi $phi
fi

