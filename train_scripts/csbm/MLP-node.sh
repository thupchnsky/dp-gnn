nk=$1
device=$2
phi=$3
epsilon=16
epoch=100

# cSBM

OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=$device python train.py mlp-dp \
    --dataset cSBM \
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
    --device cuda \
    --project MLP/node/csbm/$phi/$epsilon/ \
    --csbm_phi $phi

###############################################
# facebook

# python train.py dpdgc-kndp \
#     --dataset facebook \
#     --epsilon $epsilon \
#     --hidden-dim 64 \
#     --dropout 0.0 \
#     --encoder-epochs $epoch \
#     --encoder-lr 1e-3 \
#     --epochs $epoch \
#     --learning-rate 1e-3 \
#     --repeats 10 \
#     --batch-size 256 \
#     --device $device \
#     --norm-scale 1e-8 \
#     --nk $nk \
#     --project DPDGC/kneighbor/$nk/$epsilon/

# ###############################################
# # cora

# python train.py dpdgc-kndp \
#     --dataset cora \
#     --epsilon $epsilon \
#     --hidden-dim 64 \
#     --dropout 0.0 \
#     --encoder-epochs $epoch \
#     --encoder-lr 1e-3 \
#     --epochs $epoch \
#     --learning-rate 1e-3 \
#     --repeats 10 \
#     --batch-size 64 \
#     --device $device \
#     --norm-scale 1e-8 \
#     --nk $nk \
#     --project DPDGC/kneighbor/low_train/$nk/$epsilon/

###############################################
# pubmed

# python train.py dpdgc-kndp \
#     --dataset pubmed \
#     --epsilon $epsilon \
#     --hidden-dim 64 \
#     --dropout 0.0 \
#     --encoder-epochs $epoch \
#     --encoder-lr 1e-3 \
#     --epochs $epoch \
#     --learning-rate 1e-3 \
#     --repeats 10 \
#     --batch-size 256 \
#     --device $device \
#     --norm-scale 1e-8 \
#     --nk $nk \
#     --project DPDGC/kneighbor/low_train/$nk/$epsilon/
    
###############################################
# photo

# python train.py dpdgc-kndp \
#     --dataset photo \
#     --epsilon $epsilon \
#     --hidden-dim 64 \
#     --dropout 0.0 \
#     --encoder-epochs $epoch \
#     --encoder-lr 1e-3 \
#     --epochs $epoch \
#     --learning-rate 1e-3 \
#     --repeats 10 \
#     --batch-size 64 \
#     --device $device \
#     --norm-scale 1e-8 \
#     --nk $nk \
#     --project DPDGC/kneighbor/low_train/$nk/$epsilon/
    
    
###############################################
# computers

# python train.py dpdgc-kndp \
#     --dataset computers \
#     --epsilon $epsilon \
#     --hidden-dim 64 \
#     --dropout 0.0 \
#     --encoder-epochs $epoch \
#     --encoder-lr 1e-3 \
#     --epochs $epoch \
#     --learning-rate 1e-3 \
#     --repeats 10 \
#     --batch-size 64 \
#     --device $device \
#     --norm-scale 1e-8 \
#     --nk $nk \
#     --project DPDGC/kneighbor/low_train/$nk/$epsilon/

###############################################
# squirrel

# python train.py dpdgc-kndp \
#     --dataset squirrel \
#     --epsilon $epsilon \
#     --hidden-dim 64 \
#     --dropout 0.0 \
#     --encoder-epochs $epoch \
#     --encoder-lr 1e-3 \
#     --epochs $epoch \
#     --learning-rate 1e-3 \
#     --repeats 10 \
#     --batch-size 64 \
#     --device $device \
#     --norm-scale 1e-8 \
#     --nk $nk \
#     --project DPDGC/kneighbor/$nk/$epsilon/

###############################################
# chameleon

# python train.py dpdgc-kndp \
#     --dataset chameleon \
#     --epsilon $epsilon \
#     --hidden-dim 64 \
#     --dropout 0.0 \
#     --encoder-epochs $epoch \
#     --encoder-lr 1e-3 \
#     --epochs $epoch \
#     --learning-rate 1e-3 \
#     --repeats 10 \
#     --batch-size 64 \
#     --device $device \
#     --norm-scale 1e-8 \
#     --nk $nk \
#     --project DPDGC/kneighbor/$nk/$epsilon/



