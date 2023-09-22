device=$1

for dataset in facebook cora pubmed photo computers squirrel chameleon
do
    python train.py sage-inf \
    --dataset $dataset \
    --base_layers 2 \
    --head_layers 1 \
    --mp_layers 2 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --batch_norm True \
    --epochs 100 \
    --batch_size full \
    --device $device \
    --project SAGE/nopriv
done