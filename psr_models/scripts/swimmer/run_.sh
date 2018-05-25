#!/bin/bash
for dim in 10 15 20 30 50
do
    echo $dim
    python -m psr_models.tests.call_test --method gym_model --env Swimmer-v1 --render 1 --monitor swim --numtrajs 100 --batch 100 --len 200 --reg 1e-5 --rstep 1e-5 --fext nystrom --refine 0 --gen train --kw 50 --dim ${dim} &> results/CartPole-v0/logs/log_dim${dim}.txt &
done


