# conda activate tf
# export PYTHONPATH=`pwd`/code
python src/process_training.py train >& log/train_log &
