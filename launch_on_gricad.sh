#!/bin/bash

#OAR -n Experiment_3-1_ADC_2d_ddpm_simplex_noise
#OAR -l /nodes=1/gpu=1,walltime=48:00:00
#OAR --stdout Experiment_3-1_ADC_2d_ddpm_simplex_noise.out
#OAR --stderr Experiment_3-1_ADC_2d_ddpm_simplex_noise.err
#OAR --project pr-gin5_aini
#OAR -p gpumodel='V100'

source ../environments/ddpm_env/bin/activate

python3 Experiment_3-1_ADC_2d_ddpm_simplex_noise.py