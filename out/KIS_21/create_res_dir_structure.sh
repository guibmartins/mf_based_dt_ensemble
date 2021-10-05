#!/bin/bash
mkdir k_2 k_2/stats
for baseline in knn opf; do
    mkdir k_2/$baseline
    for alg in nmf pmf svd; do
        mkdir k_2/$baseline/$alg
        for dataset in blood cancer cmc digits iris; do
            mkdir k_2/$baseline/$alg/$dataset
        done
    done
done
