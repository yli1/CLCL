EXP_ID=main_proposed_scan.a

python attention_visualization.py \
--hide_switch \
--hide_bar \
--input_folder attention_eval1_1 \
--output_folder attention_eval1_1_vis \
--experiment_id ${EXP_ID}

python attention_visualization.py \
--hide_switch \
--hide_bar \
--input_folder attention_eval2_1 \
--output_folder attention_eval2_1_vis \
--experiment_id ${EXP_ID}

python attention_visualization.py \
--hide_switch \
--hide_bar \
--input_folder attention_eval3_1 \
--output_folder attention_eval3_1_vis \
--experiment_id ${EXP_ID}