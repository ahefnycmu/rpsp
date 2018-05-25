#!/bin/bash
for dim in 10 15 20 30
do
    echo $dim
    python -m psr_models.tests.call_test --method gym_model --env CartPole-v0 --render 0 --fut 7 --past 10 --numtrajs 1000 --batch 1000 --len 200 --reg 1e-7 --rstep 1e-5 --fext nystrom --Hdim 1000 --refine 20 --gen last --kw 70 --dim ${dim} --lr 1e-3 --wpred 0.1 --wnext 0.5 --blindN 2000 --ntrain 30 &> results/CartPole-v0/logs/log_dim${dim}.txt &
done


