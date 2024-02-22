CLIENT_NUM=$1
SAMPLE_NUM=$2
WORKER_NUM=$3
ROUND=$4
BATCH_NUM=$5
LR=$6

PROCESS_NUM=`expr $WORKER_NUM + 1`

mpirun -n $PROCESS_NUM python3 main.py \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $SAMPLE_NUM \
  --gpu_worker_num $WORKER_NUM \
  --round $ROUND \
  --batch_size $BATCH_NUM \
  --lr $LR