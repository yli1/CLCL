EXP_ID=logs/main_proposed_scan.a
INPUT_DIR=${EXP_ID}/params/100

python tsne/tsne_visualization.py \
--input_file ${INPUT_DIR}/primitive.txt \
--dict_file ${EXP_ID}/voc_dict.txt \
--load_tsne_output \
--separate_plot \
--max_elements 0 \
--output_prefix outputs/primitive_visualization

python tsne/tsne_visualization.py \
--input_file ${INPUT_DIR}/function.txt \
--dict_file ${EXP_ID}/voc_dict.txt \
--load_tsne_output \
--separate_plot \
--max_elements 0 \
--output_prefix outputs/function_visualization

python tsne/tsne_visualization.py \
--input_file ${INPUT_DIR}/prediction.txt \
--dict_file ${EXP_ID}/act_dict.txt \
--load_tsne_output \
--separate_plot \
--transpose \
--initial_stage_size 6 \
--max_elements 0 \
--output_prefix outputs/prediction_visualization