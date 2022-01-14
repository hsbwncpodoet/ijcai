#!/bin/bash

#mixtures=(0.0 0.1 0.3 0.5 0.7 0.9 1.0)
mixtures=(2)
#basepath="./3x3/mixed_action_affine_"
basepath="./3x3/heuristic_2_viol_"
#basepath="./2x2/mixed_action_single_raw/mixed_action_single_raw_"
savepath="./3x3/heuristic_2_viol_2/"
vset="2"
world="4"
categories="5"

## 1. Policy and Reward Extraction
for pct in ${mixtures[@]}; do
    ## Extract reward will save as a *.pkl.rewards file
    python ep_plot.py --mode extract_reward \
        --inputs ${basepath}${pct}/exp_*.pkl \
        --groupings exp \
        --world ${world} \
        --categories ${categories}
    ## Extract reward will save as a *.pkl.policy file
    python ep_plot.py --mode extract_policies \
        --inputs ${basepath}${pct}/exp_*.pkl \
        --groupings exp \
        --world ${world} \
        --categories ${categories}
done

## 2. Evaluations
for pct in ${mixtures[@]}; do
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.policy \
        --evalmetric dgs \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world} \
        --categories ${categories}
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.policy \
        --evalmetric dpv \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world} \
        --categories ${categories}
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.policy \
        --evalmetric sgs \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world} \
        --categories ${categories}
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.policy \
        --evalmetric spv \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world} \
        --categories ${categories}
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.rewards \
        --evalmetric rew \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world} \
        --categories ${categories}
done

## 3. Plot the data
python ep_plot.py --mode plot \
    --inputs \
    ${basepath}2/dat.dgs \
    --world ${world} \
    --categories ${categories} \
    --ylo -0.1 --yhi 1.1 --scale 0.1 \
    --xlabel Steps --ylabel "Goal Success" \
    --plottitle "Goal Success" \
    --saveplot ${savepath}deterministic_goal_success.per_step.pdf

python ep_plot.py --mode plot \
    --inputs \
    ${basepath}2/dat.sgs \
    --world ${world} \
    --categories ${categories} \
    --ylo -0.1 --yhi 1.1 --scale 0.1 \
    --xlabel Steps --ylabel "Goal Success" \
    --plottitle "Goal Success" \
    --saveplot ${savepath}stochastic_goal_success.per_step.pdf

python ep_plot.py --mode plot \
    --inputs \
    ${basepath}2/dat.dpv \
    --world ${world} \
    --categories ${categories} \
    --ylo -0.1 --yhi 4 --scale 0.1 \
    --xlabel Steps --ylabel "Violations" \
    --plottitle "Violations" \
    --saveplot ${savepath}deterministic_policy_violations.per_step.pdf

python ep_plot.py --mode plot \
    --inputs \
    ${basepath}2/dat.spv \
    --world ${world} \
    --categories ${categories} \
    --ylo -0.1 --yhi 1.1 --scale 0.1 \
    --xlabel Steps --ylabel "Violations" \
    --plottitle "Violations" \
    --saveplot ${savepath}stochastic_policy_violations.per_step.pdf

for pct in ${mixtures[@]}; do
    python ep_plot.py --mode plot_reward \
    --inputs \
    ${basepath}${pct}/dat.rew \
    --world ${world} \
    --categories ${categories} \
    --xlabel Steps \
    --ylabel "Reward" \
    --plottitle "Reward" \
    --saveplot ${savepath}reward.${pct}.per_step.pdf
done
