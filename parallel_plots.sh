## Step 1: Policy Extraction
## The --groupings option takes the paths in --inputs and groups them according to
## keywords.
##
## So if you have ../../action_1.pkl, ../../action_2.pkl, ....
## and you also have ../../eval_1.pkl, ../../eval_2.pkl, ....
## then you can specify --groupings action eval
## to create a group for all "action" data, and another for all "eval" data.
##
## The --lastonly option specifies that policies should only be extracted from the
## last timestep.
##
python ep_plot.py --mode extract_policies --inputs ./exp_*.pkl --groupings exp --lastonly

## Step 2: Evaluations
##   dgs = deterministic goal success
##   dpv = deterministic policy violations
##   sgs = stochastic goal success
##   spv = stochastic policy violations
python ../../../../ep_plot.py --mode single_group_metric --inputs exp_*.pkl.policy --evalmetric dgs --dataset pcb
python ../../../../ep_plot.py --mode single_group_metric --inputs exp_*.pkl.policy --evalmetric dpv --dataset pcb
python ../../../../ep_plot.py --mode single_group_metric --inputs exp_*.pkl.policy --evalmetric sgs --dataset pcb
python ../../../../ep_plot.py --mode single_group_metric --inputs exp_*.pkl.policy --evalmetric spv --dataset pcb

## Step 3: Plotting
## 
## Plot the evaluations from step 2.
##
python ../../../../ep_plot.py --mode plot --inputs pcb.dgs --ylo -0.1 --yhi 1.1 --scale 0.1 --xlabel Episodes --ylabel "Goal Success" --plottitle "Goal Success" --saveplot goal_success.per_episode.pdf
python ../../../../ep_plot.py --mode plot --inputs pcb.dpv --ylo -0.1 --yhi 4 --scale 0.1 --xlabel Episodes --ylabel "Violations" --plottitle "Violations" --saveplot policy_violations.per_episode.pdf
