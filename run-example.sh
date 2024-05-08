#!/bin/bash
START_SEED=1
END_SEED=11
EXPERIMENT="NSGA3_EXAMPLE"
POPULATION_SIZE=647
MUTATION_PROB=0.15
MUTATION_SIGMA=0.14
XOVER_POINTS=3
XOVER_PROB=0.9

for (( SEED=$START_SEED; SEED<=$END_SEED; SEED++ ))
do
    echo "Running with seed=$SEED"
    python optim.py --verbose --experiment $EXPERIMENT --seed $SEED --objs-file objectives.conf.json --algorithm NSGA3 --ref-dirs-method energy --ref-dirs-n-points $POPULATION_SIZE --max-gens 100 --population-size $POPULATION_SIZE --mutation-prob $MUTATION_PROB --mutation-sigma $MUTATION_SIGMA --xover-points $XOVER_POINTS --xover-prob $XOVER_PROB
done
