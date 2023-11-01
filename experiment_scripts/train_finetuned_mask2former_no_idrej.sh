#block(name=finetune_m2f,threads=8,gpus=1,memory=10000,hours=6)
  SUBTASK_ID=62
  seq_num=$((SUBTASK_ID-1))
  seq_name="seq_${seq_num}"
  
  echo "Running subtask $SUBTASK_ID"
  echo "frame index: $seq_num"

  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config "_results/logs/runs/best_delta_deeplab_depth_filtered/test_frame_10/20230731-175257/config.yaml" \
  --pretrained ""\
  --dataset-mode "all_frames_window" \
  --dataset-num-workers 6 \
  --log-dir "/scratchdata/csmitt/bup20_nerf_inference/no_idrej_mask2former_finetuned"\
  --dataset-center-idx "${seq_num}" \
  --exp-name "${seq_name}"\
  --save-preds \
  --inst-outlier-rejection ""\
  --inst-num-dilations 1\
  --inst-conf-enable \
  --sem-conf-enable \
  --max-depth 1.2\
  --inst-segment-reg-weight 0.1\
  --inst-segment-reg-epoch-start 101 \
  --load-modes imgs semantics instance ../../preds_mask2former_finetuned inst_conf sem_conf \
  --class-labels bg red yellow green mixed_red mixed_yellow \
  --valid-every 400