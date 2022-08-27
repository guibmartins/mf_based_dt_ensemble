#!/bin/bash
cd /mnt/Dados/Guilherme/Doutorado/Libraries/PycharmProjects/Ensemble_OPF_MF/
# arg baseline: 'knn' or 'opf'
baseline=$1
# arg_1: iterations; arg_2: n_learners; arg_3: n_factors (MF); arg_4: n_folds (models); arg_5: custom seed
for mf in nmf pmf svd; do
    for data in blood cancer iris cmc digits; do
        /home/guilherme/Programs/anaconda3/envs/ensemble_opf_mf/bin/python3.8 main_experiments.py $baseline $mf $data 20 5 2 5 1902
    done
done
