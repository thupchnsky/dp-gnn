device=$1

for phi in -1.0 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1.0
do
    source DPDGC-kneighbor.sh 1 $device $phi
    source DPDGC-node.sh 1 $device $phi
    source DPDGC-nopriv.sh 1 $device $phi
    source DPDGC-edge.sh 1 $device $phi
    source GAP-edge.sh 1 $device $phi
    source GAP-node.sh 1 $device $phi
    source GAP-nopriv.sh 1 $device $phi
    source MLP-node.sh 1 $device $phi
    source MLP-nopriv.sh 1 $device $phi
done