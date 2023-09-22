device=$1

for phi in -1.0 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1.0
do
    source ./train_scripts/csbm/DPDGC-kneighbor.sh 1 $device $phi
    source ./train_scripts/csbm/DPDGC-node.sh 1 $device $phi
    source ./train_scripts/csbm/DPDGC-nopriv.sh 1 $device $phi
    source ./train_scripts/csbm/DPDGC-edge.sh 1 $device $phi
    source ./train_scripts/csbm/GAP-edge.sh 1 $device $phi
    source ./train_scripts/csbm/GAP-node.sh 1 $device $phi
    source ./train_scripts/csbm/GAP-nopriv.sh 1 $device $phi
    source ./train_scripts/csbm/MLP-node.sh 1 $device $phi
    source ./train_scripts/csbm/MLP-nopriv.sh 1 $device $phi
done
