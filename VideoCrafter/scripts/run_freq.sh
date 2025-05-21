#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"
source ~/anaconda3/etc/profile.d/conda.sh

conda deactivate
conda activate videocrafter

name="base_512_v2_freq_low"

ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_512_v2_freq.0.yaml'

prompt_file="prompts/test_prompts.txt"
res_dir="results"

python3 scripts/evaluation/inference_freq.py \
--seed 123 \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--fps 28 \
--lib 'func_freq'
