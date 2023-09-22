nk=$1
device=$2
epsilon=16
epoch=100
###############################################
# facebook

python train.py dpdgc-kndp \
    --dataset facebook \
    --epsilon $epsilon \
    --hidden-dim 64 \
    --dropout 0.0 \
    --encoder-epochs $epoch \
    --encoder-lr 1e-3 \
    --epochs $epoch \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size 256 \
    --device $device \
    --norm-scale 1e-8 \
    --nk $nk \
    --project DPDGC/kneighbor/$nk/$epsilon/

# ###############################################
# # cora

python train.py dpdgc-kndp \
    --dataset cora \
    --epsilon $epsilon \
    --hidden-dim 64 \
    --dropout 0.0 \
    --encoder-epochs $epoch \
    --encoder-lr 1e-3 \
    --epochs $epoch \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size 64 \
    --device $device \
    --norm-scale 1e-8 \
    --nk $nk \
    --project DPDGC/kneighbor/$nk/$epsilon/

###############################################
# pubmed

python train.py dpdgc-kndp \
    --dataset pubmed \
    --epsilon $epsilon \
    --hidden-dim 64 \
    --dropout 0.0 \
    --encoder-epochs $epoch \
    --encoder-lr 1e-3 \
    --epochs $epoch \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size 256 \
    --device $device \
    --norm-scale 1e-8 \
    --nk $nk \
    --project DPDGC/kneighbor/$nk/$epsilon/
    
###############################################
# photo

python train.py dpdgc-kndp \
    --dataset photo \
    --epsilon $epsilon \
    --hidden-dim 64 \
    --dropout 0.0 \
    --encoder-epochs $epoch \
    --encoder-lr 1e-3 \
    --epochs $epoch \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size 64 \
    --device $device \
    --norm-scale 1e-8 \
    --nk $nk \
    --project DPDGC/kneighbor/$nk/$epsilon/
    
    
###############################################
# computers

python train.py dpdgc-kndp \
    --dataset computers \
    --epsilon $epsilon \
    --hidden-dim 64 \
    --dropout 0.0 \
    --encoder-epochs $epoch \
    --encoder-lr 1e-3 \
    --epochs $epoch \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size 64 \
    --device $device \
    --norm-scale 1e-8 \
    --nk $nk \
    --project DPDGC/kneighbor/$nk/$epsilon/

###############################################
# squirrel

python train.py dpdgc-kndp \
    --dataset squirrel \
    --epsilon $epsilon \
    --hidden-dim 64 \
    --dropout 0.0 \
    --encoder-epochs $epoch \
    --encoder-lr 1e-3 \
    --epochs $epoch \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size 64 \
    --device $device \
    --norm-scale 1e-8 \
    --nk $nk \
    --project DPDGC/kneighbor/$nk/$epsilon/

###############################################
# chameleon

python train.py dpdgc-kndp \
    --dataset chameleon \
    --epsilon $epsilon \
    --hidden-dim 64 \
    --dropout 0.0 \
    --encoder-epochs $epoch \
    --encoder-lr 1e-3 \
    --epochs $epoch \
    --learning-rate 1e-3 \
    --repeats 10 \
    --batch-size 64 \
    --device $device \
    --norm-scale 1e-8 \
    --nk $nk \
    --project DPDGC/kneighbor/$nk/$epsilon/



