#!/bin/bash

# Tell the system the resources you need. Adjust the numbers according to your need, e.g.
#SBATCH --gres=gpu:1 --cpus-per-task=4 --mail-type=ALL

#If you use Anaconda, initialize it
. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate retinanet

# cd your your desired directory and execute your program, e.g.
cd $HOME/HKU-DASC7606-A1/
python train.py --coco_path ./data --output_path ./output_101_au_res --depth 101 --epochs 30
