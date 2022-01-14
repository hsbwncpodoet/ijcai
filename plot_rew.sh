mixtures=(0.0 0.1 0.3 0.5 0.7 0.9 1.0)

savepath="./3x3/doubled/"
basepath="./3x3/doubled/mixed_action_single_raw_"
#basepath="./3x3/mixed_action_single_affine/mixed_action_affine_"
#basepath="./3x3/mixed_action_single_raw/mixed_action_raw_"
#basepath="./3x3/doubled/mixed_action_single_raw_"
vset="1"
world="4"
categories="5"

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
