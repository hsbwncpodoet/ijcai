mixtures=(0.0 0.1 0.3 0.5 0.7 0.9 1.0)
#mixtures=(1.0)

basepath="./3x3/mixed_action_affine_"
vset="2"
world="5"






for pct in ${mixtures[@]}; do
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.policy \
        --evalmetric dgs \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world}
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.policy \
        --evalmetric dpv \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world}
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.policy \
        --evalmetric sgs \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world}
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.policy \
        --evalmetric spv \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world}
    python ep_plot.py --mode single_group_metric \
        --inputs ${basepath}${pct}/exp_*.pkl.rewards \
        --evalmetric rew \
        --dataset ${basepath}${pct}/dat \
        --violation_set ${vset} \
        --world ${world}
done
