#!/bin/bash

export PROJECT_PATH="/home/ray/Chain-of-Embedding/"
export CUDA_VISIBLE_DEVICES="0"

model_name="llama3.2-1b-instruct"
dataset_list=(test)

for i in ${dataset_list[*]}; do
    /opt/miniconda/envs/coeeval/bin/python main.py --model_name $model_name \
                        --dataset "$i" \
                        --print_model_parameter \
                        --save_output \
                        --save_hidden_states \
                        --save_coe_score \
                        --save_coe_figure
done
