#!/usr/bin/env bash

#SBATCH --gres gpu
#SBATCH --partition gpu,gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --time 0-18:00  # D-H:M
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

. ~/load_ap.src

./compare_field_types.py $@

