#block(name=panoli_conf,threads=8,gpus=1,memory=10000,hours=6)
  SUBTASK_ID=271

  seq_num=$((SUBTASK_ID-1))
  seq_name="seq_${seq_num}"
  
  echo "Running subtask $SUBTASK_ID"
  echo "frame index: $seq_num"


  _NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config configs/bup20/panoptic_lifting_app.yaml \
  --dataset-num-workers 6 \
  --dataset-center-idx "${SUBTASK_ID-1}"\
  --log-dir "/scratchdata/csmitt/bup20_nerf_inference/panoli_mask2former_sequence"\
  --dataset-center-idx "${seq_num}" \
  --exp-name "${seq_name}"\
  --save-preds \
  --inst-conf-enable \
  --sem-conf-enable \
  --dataset-mode "all_frames_window" \
  --load-modes imgs semantics instance preds_mask2former inst_conf sem_conf \
  --valid-every 400


  