device=$1
epoch=100

##############################################

## facebook 
python train.py dpdgc-inf \
--dataset facebook \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size full \
--device $device \
--norm-scale 1e-8 \
--project DPDGC/nopriv/

# ################################################

## cora 
python train.py dpdgc-inf \
--dataset cora \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size full \
--device $device \
--norm-scale 1e-8 \
--project DPDGC/nopriv/

################################################

## pubmed 
python train.py dpdgc-inf \
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
--project DPDGC/nopriv/

###############################################

## photo 
python train.py dpdgc-inf \
--dataset photo \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size full \
--device $device \
--norm-scale 1e-8 \
--project DPDGC/nopriv/

###############################################

## computers 
python train.py dpdgc-inf \
--dataset computers \
--hidden-dim 64 \
--dropout 0.5 \
--encoder-epochs $epoch \
--encoder-lr 1e-3 \
--epochs $epoch \
--learning-rate 1e-3 \
--repeats 10 \
--batch-size full \
--device $device \
--norm-scale 1e-8 \
--project DPDGC/nopriv/

################################################

## squirrel 
python train.py dpdgc-inf \
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
--project DPDGC/nopriv/

################################################

## chameleon 
python train.py dpdgc-inf \
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
--project DPDGC/nopriv/


