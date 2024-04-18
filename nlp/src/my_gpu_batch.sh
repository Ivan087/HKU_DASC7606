#!/bin/bash

# Tell the system the resources you need. Adjust the numbers according to your need, e.g.
#SBATCH --gres=gpu:1 --cpus-per-task=4 --mail-type=ALL

#If you use Anaconda, initialize it
. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp_env

# cd your your desired directory and execute your program, e.g.
cd $HOME/HKU-DASC7606-A2/

# python eval_fewshot_multigpu.py --output_path output --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Easy-validation.jsonl" --device_id "0,1" --model microsoft/phi-2 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output

# python eval_fewshot.py --output_path output_cha_15_base --prefix "base" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Challenge-validation.jsonl" --device_id "0,1" --model microsoft/phi-1_5 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output_cha_15_base

# python eval_fewshot.py --output_path output_cha_15_pre --prefix "pre" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Challenge-validation.jsonl" --device_id "0,1" --model microsoft/phi-1_5 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output_cha_15_pre

# python eval_fewshot.py --output_path output_cha_15_cot --prefix "cot" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Challenge-validation.jsonl" --device_id "0,1" --model microsoft/phi-1_5 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output_cha_15_cot

# python eval_fewshot_multigpu.py --output_path output_easy_2_base --prefix "base" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Easy-validation.jsonl" --device_id "0,1" --model microsoft/phi-2 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output_easy_2_base 

# python eval_fewshot_multigpu.py --output_path output_easy_2_pre --prefix "pre" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Easy-validation.jsonl" --device_id "0,1" --model microsoft/phi-2 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output_easy_2_pre 

# python eval_fewshot_multigpu.py --output_path output_easy_2_cot --prefix "cot" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Easy-validation.jsonl" --device_id "0,1" --model microsoft/phi-2 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output_easy_2_cot 

# python eval_fewshot_multigpu.py --output_path output_cha_2_base --prefix "base" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Challenge-validation.jsonl" --device_id "0,1" --model microsoft/phi-2 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output_cha_2_base 

# python eval_fewshot_multigpu.py --output_path output_cha_2_pre --prefix "pre" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Challenge-validation.jsonl" --device_id "0,1" --model microsoft/phi-2 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output_cha_2_pre 

# python eval_fewshot_multigpu.py --output_path output_cha_2_cot --prefix "cot" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Challenge-validation.jsonl" --device_id "0,1" --model microsoft/phi-2 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
# python acc.py --prediction_path output_cha_2_cot 

## Test
python eval_fewshot_multigpu.py --output_path Easy_test --prefix "base" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Easy-test.jsonl" --device_id "0,1" --model microsoft/phi-2 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
python acc.py --prediction_path Easy_test 

python eval_fewshot_multigpu.py --output_path Challenge_test --prefix "base" --overwrite True --prompt_type "v2.0" --N 8 --data_path "data/ARC-Challenge-test.jsonl" --device_id "0,1" --model microsoft/phi-2 --embedder BAAI/bge-small-en-v1.5 --start_index 0 --end_index 9999 --max_len 1024  --top_k True --top_k_reverse False
python acc.py --prediction_path Challenge_test 