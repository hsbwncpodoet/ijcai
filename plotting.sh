python multimodal_feedback.py \
    --mode plot \
    --inputs data/multimodal_feedback_request/full_goal_violations_0.pkl \
             data/multimodal_feedback_request/full_goal_violations_1.pkl \
             data/multimodal_feedback_request/full_goal_violations_2.pkl \
             data/multimodal_feedback_request/full_goal_violations_3.pkl \
             data/multimodal_feedback_request/full_goal_violations_4.pkl \
             data/multimodal_feedback_request/traj_goal_violations_0.pkl \
             data/multimodal_feedback_request/traj_goal_violations_1.pkl \
             data/multimodal_feedback_request/traj_goal_violations_2.pkl \
             data/multimodal_feedback_request/traj_goal_violations_3.pkl \
             data/multimodal_feedback_request/traj_goal_violations_4.pkl \
             data/multimodal_always_feedback/human_choice_0.pkl \
             data/multimodal_always_feedback/human_choice_1.pkl \
             data/multimodal_always_feedback/human_choice_2.pkl \
             data/multimodal_always_feedback/human_choice_3.pkl \
             data/multimodal_always_feedback/human_choice_4.pkl \
             data/multimodal_always_feedback/action_only_0.pkl \
             data/multimodal_always_feedback/action_only_1.pkl \
             data/multimodal_always_feedback/action_only_2.pkl \
             data/multimodal_always_feedback/action_only_3.pkl \
             data/multimodal_always_feedback/action_only_4.pkl \
             data/multimodal_no_batch/scalar_only_0.pkl \
             data/multimodal_no_batch/scalar_only_1.pkl \
             data/multimodal_no_batch/scalar_only_2.pkl \
             data/multimodal_prepop_overfit_batch_100/baseline.pkl \
    --groupings full traj human_choice action scalar baseline \
    --hide --dataset blah
