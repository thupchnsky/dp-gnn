device=$1

for dataset in facebook cora pubmed photo computers squirrel chameleon
do
    python train.py mlp \
    --dataset $dataset \
    --num_layers 3 \
    --hidden_dim 64 \
    --activation selu \
    --optimizer adam \
    --learning_rate 1e-3 \
    --repeats 10 \
    --epochs 100 \
    --batch_size full \
    --dropout 0.5 \
    --device $device \
    --project MLP/nopriv
done

