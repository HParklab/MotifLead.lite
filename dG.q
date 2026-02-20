#!/bin/bash
#SBATCH -p A5000_node
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu2
#SBATCH -c 10
#SBATCH -N 1
##SBATCH --mem=40g
#SBATCH -o dG.log

python devel_dG.py
