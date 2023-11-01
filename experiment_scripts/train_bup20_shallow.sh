#block(name=shallow_grid,threads=8,gpus=1,memory=10000,subtasks=100)
  paths=( `echo _results/logs/runs/best_delta_deeplab_depth_filtered/test_frame_*/*` )
  #store length of paths array in a variable
  num_paths=${#paths[@]}
  
  path=${paths[$((($SUBTASK_ID-1) ))]}
  frame_name="$(cut -d'/' -f5 <<<"${path}")"
  
  echo "Running subtask $SUBTASK_ID"
  echo "frame index: $frame_name"

  finest_scale=0.001
  num_lods=15
  capacity=15
  delta_capacity=10

  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config "${path}/config.yaml" \
  --dataset-num-workers 6 \
  --log-dir "_results/logs/runs/best_models/best_delta_mask2former_fscale_${finest_scale}_lods_${num_lods}_cap_${capacity}_dcap_${delta_capacity}"\
  --exp-name "${frame_name}"\
  --inst-outlier-rejection \
  --inst-num-dilations 1\
  --inst-conf-enable \
  --sem-conf-enable \
  --max-depth 1.2\
  --inst-segment-reg-weight 0.1\
  --inst-segment-reg-epoch-start 701 \
  --load-modes imgs semantics instance preds_mask2former inst_conf sem_conf \
  --finest-scale "${finest_scale}"\
  --num-lods "${num_lods}"\
  --capacity-log-2 "${capacity}"\
  --delta-capacity-log-2 "${delta_capacity}"\