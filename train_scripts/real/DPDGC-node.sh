device=$1
epoch=100
epsilon=16

###############################################
# facebook

python ./train.py dpdgc-ndp \
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
    --max_degree 100 \
    --project DPDGC/ndp/$epsilon/

###############################################
# cora

python ./train.py dpdgc-ndp \
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
    --max_degree 100 \
    --project DPDGC/ndp/$epsilon/

###############################################
# pubmed

python ./train.py dpdgc-ndp \
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
    --max_degree 100 \
    --project DPDGC/ndp/$epsilon/
    
###############################################
# photo

python ./train.py dpdgc-ndp \
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
    --max_degree 100 \
    --project DPDGC/ndp/$epsilon/
    
    
###############################################
# computers

python ./train.py dpdgc-ndp \
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
    --max_degree 100 \
    --project DPDGC/ndp/$epsilon/

###############################################
# squirrel

python ./train.py dpdgc-ndp \
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
    --max_degree 100 \
    --project DPDGC/ndp/$epsilon/

###############################################
# chameleon

python ./train.py dpdgc-ndp \
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
    --max_degree 100 \
    --project DPDGC/ndp/$epsilon/



