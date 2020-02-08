#!/usr/bin/env bash

ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

MYDIR=logs/${ID}
mkdir -p ${MYDIR}
cp ${ABS_PATH} ${MYDIR}

TF_FORCE_GPU_ALLOW_GROWTH=true \
CUDA_VISIBLE_DEVICES=1 \
python -u continual_learning_main.py \
--experiment_id ${ID} \
--data_name continual \
--continual_learning \
--continual_all_params \
--stages 100 \
--remove_prediction_bias \
--data_dir data/data_translate \
--use_stage_data \
--model_name continual_normal \
--random_seed 52 \
--batch_size 512 \
--test_batch_size 1000 \
--switch_temperature 0.1 \
--attention_temperature 1 \
--num_units 64 \
--epochs 5000 \
--continual_epochs 1000 \
--learning_rate 0.01 \
--max_gradient_norm 1.0 \
--use_input_length \
--use_embedding \
--embedding_size 64 \
--function_embedding_size 8 \
--bidirectional_encoder \
--random_batch \
--decay_steps 100 \
--remove_switch \
--single_representation \
--use_decoder_input \
--content_noise_coe 0.1 \
--sample_wise_content_noise \
--masked_attention \
--random_random \
| tee ${MYDIR}/log.txt
