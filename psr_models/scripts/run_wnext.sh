#!/bin/bash
for wnext in 1.0 0.0 0.01 0.1 0.5
do
    echo $wnext
    python -m psr_models.tests.call_test --method gym_model --env CartPole-v0 --render 0 --fut 7 --past 10 --numtrajs 1000 --batch 1000 --len 200 --reg 1e-7 --rstep 1e-5 --fext nystrom --Hdim 1000 --refine 20 --gen last --kw 70 --dim 10 --lr 1e-3 --wpred 0.1 --wnext ${wnext} --blindN 2000 --ntrain 30 &> results/CartPole-v0/logs/log_wnext${wnext}.txt &
done


