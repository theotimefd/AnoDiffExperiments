#!/bin/bash

#OAR -n Experiment_3-1_compute_dice
#OAR -l /nodes=1/gpu=1,walltime=10:00:00
#OAR --stdout Experiment_3-1_compute_dice.out
#OAR --stderr Experiment_3-1_compute_dice.err
#OAR --project pr-gin5_aini
#OAR -p gpumodel='V100'

source ../environments/ddpm_env/bin/activate

python3 Experiment_3_1_compute_dice.py