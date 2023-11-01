#!/bin/bash 

__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
WISP_HEADLESS=1 \
python main_interactive.py \
--config "config/bup20/best.yaml" \
--dataset-path "datasets/BUP20"
--pretrained ""\
--dataset-num-workers 6 \
--log-dir "_results/logs/runs/bup20"\
--dataset-center-idx "10" \
--exp-name "seq_10"\
--save-preds \
--inst-outlier-rejection \
--inst-num-dilations 1\
--inst-conf-enable \
--sem-conf-enable \
--max-depth 1.2\
--inst-segment-reg-weight 0.1\
--inst-segment-reg-epoch-start 101 \
--load-modes imgs semantics instance ../../preds_mask2former_finetuned inst_conf sem_conf \
--class-labels bg red yellow green mixed_red mixed_yellow \
--valid-every 200