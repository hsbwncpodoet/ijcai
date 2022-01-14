mixtures=(0.0 0.1 0.3 0.5 0.7 0.9 1.0)

savepath="./3x3/mixed_action_single_raw/violating_3/"
basepath="./3x3/mixed_action_single_raw/violating_3/mixed_action_single_raw_"
#basepath="./3x3/mixed_action_single_affine/mixed_action_affine_"
#basepath="./3x3/mixed_action_single_raw/mixed_action_raw_"
#basepath="./3x3/doubled/mixed_action_single_raw_"
vset="1"
world="4"
categories="5"

python ep_plot.py --mode plot \
    --inputs \
    ${basepath}0.0/dat.dgs \
    ${basepath}0.1/dat.dgs \
    ${basepath}0.3/dat.dgs \
    ${basepath}0.5/dat.dgs \
    ${basepath}0.7/dat.dgs \
    ${basepath}0.9/dat.dgs \
    ${basepath}1.0/dat.dgs \
    --violation_set ${vset} \
    --categories ${categories} \
    --world ${world} \
    --ylo -0.1 --yhi 1.1 --scale 0.1 \
    --xlabel Steps --ylabel "Goal Success" \
    --plottitle "Goal Success" \
    --saveplot ${savepath}deterministic_goal_success.per_step.pdf

python ep_plot.py --mode plot \
    --inputs \
    ${basepath}0.0/dat.sgs \
    ${basepath}0.1/dat.sgs \
    ${basepath}0.3/dat.sgs \
    ${basepath}0.5/dat.sgs \
    ${basepath}0.7/dat.sgs \
    ${basepath}0.9/dat.sgs \
    ${basepath}1.0/dat.sgs \
    --violation_set ${vset} \
    --categories ${categories} \
    --world ${world} \
    --ylo -0.1 --yhi 1.1 --scale 0.1 \
    --xlabel Steps --ylabel "Goal Success" \
    --plottitle "Goal Success" \
    --saveplot ${savepath}stochastic_goal_success.per_step.pdf

python ep_plot.py --mode plot \
    --inputs \
    ${basepath}0.0/dat.dpv \
    ${basepath}0.1/dat.dpv \
    ${basepath}0.3/dat.dpv \
    ${basepath}0.5/dat.dpv \
    ${basepath}0.7/dat.dpv \
    ${basepath}0.9/dat.dpv \
    ${basepath}1.0/dat.dpv \
    --violation_set ${vset} \
    --categories ${categories} \
    --world ${world} \
    --ylo -0.1 --yhi 4 --scale 0.1 \
    --xlabel Steps --ylabel "Violations" \
    --plottitle "Violations" \
    --saveplot ${savepath}deterministic_policy_violations.per_step.pdf

python ep_plot.py --mode plot \
    --inputs \
    ${basepath}0.0/dat.spv \
    ${basepath}0.1/dat.spv \
    ${basepath}0.3/dat.spv \
    ${basepath}0.5/dat.spv \
    ${basepath}0.7/dat.spv \
    ${basepath}0.9/dat.spv \
    ${basepath}1.0/dat.spv \
    --violation_set ${vset} \
    --categories ${categories} \
    --world ${world} \
    --ylo -0.1 --yhi 1.1 --scale 0.1 \
    --xlabel Steps --ylabel "Violations" \
    --plottitle "Violations" \
    --saveplot ${savepath}stochastic_policy_violations.per_step.pdf

for pct in ${mixtures[@]}; do
    python ep_plot.py --mode plot_reward \
    --inputs \
    ${basepath}${pct}/dat.rew \
    --world ${world} \
    --violation_set ${vset} \
    --categories ${categories} \
    --xlabel Steps \
    --ylabel "Reward" \
    --plottitle "Reward" \
    --saveplot ${savepath}reward.${pct}.per_step.pdf
done
