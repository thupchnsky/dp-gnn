device=$1
epoch=100
epsilon=1
hops=2

for dataset in facebook cora pubmed photo computers squirrel chameleon
do
    python train.py gap-edp \
    --dataset $dataset \
    --epsilon $epsilon \
    --encoder_layers 2 \
    --base_layers 1 \
    --head_layers 1 \
    --combine cat \
    --hops $hops \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --batch_norm True \
    --epochs $epoch \
    --batch_size full \
    --dropout 0.5 \
    --encoder_epochs $epoch \
    --device $device \
    --project GAP/edge/$epsilon/$hops
done

