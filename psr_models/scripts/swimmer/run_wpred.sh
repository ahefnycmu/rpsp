#!/bin/bash
for wpred in 1.0 0.0 0.9 0.1 0.5
do
    echo $wpred;
    python -m psr_models.tests.call_test --method gym_model --env Swimmer-v1 --render 1 --fut 10 --past 15 --numtrajs 100 --batch 100 --len 200 --reg 1e-5 --rstep 1e-5 --fext nystrom --Hdim 1000 --refine 20 --gen boot --kw 70 --dim 20 --lr 1e-3 --wpred ${wpred} --wnext 0.5 --blindN 200 --maxtrajs 300 --ntrain 30 &> results/Swimmer-v1_gym_model/logs/log_wpred${wpred}.txt ;
done


