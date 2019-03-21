#!/bin/bash


function run() {
  echo "batch_size: $1 epoch_length: $2 num_tpus: $3"
  python collect_benchmark.py --use_tpu --batch_size=$1 --epoch_length=$2 --num_tpus=$3
}

run 16 50 8
run 32 50 8
run 64 50 8
run 128 50 8
run 16 100 8
run 16 200 8
run 16 50 4
run 16 50 2
run 16 50 1
