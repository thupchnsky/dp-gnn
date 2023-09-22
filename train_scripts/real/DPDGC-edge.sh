device=$1
epoch=100
epsilon=1
# ###############################################

## facebook

python train.py dpdgc-edp \
--dataset facebook \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--epsilon $epsilon \
--batch-size 256 \
--device $device \
--norm-scale 1e-8 \
--project DPDGC/edge/$epsilon/

# # ################################################

## cora

python train.py dpdgc-edp \
--dataset cora \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size 64 \
--device $device \
--norm-scale 1e-8 \
--epsilon $epsilon \
--project DPDGC/edge/$epsilon/

##############################################

## pubmed
python train.py dpdgc-edp \
--dataset pubmed \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size 256 \
--device $device \
--norm-scale 1e-8 \
--epsilon $epsilon \
--project DPDGC/edge/$epsilon/

# # ###############################################

## photo 

python train.py dpdgc-edp \
--dataset photo \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size 64 \
--device $device \
--norm-scale 1e-8 \
--epsilon $epsilon \
--project DPDGC/edge/$epsilon/

################################################

## computers

python train.py dpdgc-edp \
--dataset computers \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size 64 \
--device $device \
--norm-scale 1e-8 \
--epsilon $epsilon \
--project DPDGC/edge/$epsilon/

################################################

## squirrel

python train.py dpdgc-edp \
--dataset squirrel \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size 64 \
--device $device \
--norm-scale 1e-8 \
--epsilon $epsilon \
--project DPDGC/edge/$epsilon/

# ################################################

## chameleon

python train.py dpdgc-edp \
--dataset chameleon \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size 64 \
--device $device \
--norm-scale 1e-8 \
--epsilon $epsilon \
--project DPDGC/edge/$epsilon/

# ##############################################

