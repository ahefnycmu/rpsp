#!/bin/bash
for dim in 10 15 20 30
do
    echo $dim
    python -m psr_models.tests.call_test --method gym_pred --env CartPole-v0 --render 0 --numtrajs 2000 --batch 2000 --len 200 --reg 1e-4 --rstep 1e-3 --fext rff --refine 0 --gen train --kw 50 --dim ${dim} &> results/CartPole-v0/logs/log_dim${dim}.txt &
done


