set -x 
export PYTHONPATH=$PYTHONPATH:./

CUDA_VISIBLE_DEVICES=$1 python3 -m torch.distributed.launch --nproc_per_node=$2 --master_port=$3 tools/run_net.py --config-file=$4 $5
