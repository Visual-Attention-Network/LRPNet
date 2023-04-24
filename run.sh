set -x 
export PYTHONPATH=$PYTHONPATH:./

CUDA_VISIBLE_DEVICES=$1 python3 tools/run_net.py --config-file=$2 $3