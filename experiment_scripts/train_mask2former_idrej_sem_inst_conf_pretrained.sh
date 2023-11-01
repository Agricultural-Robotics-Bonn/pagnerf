#block(name=m2f_idrej_conf,threads=8,gpus=1,memory=10000,hours=6, subtasks=100)
  paths=( `echo _results/logs/runs/best_delta_deeplab_depth_filtered/test_frame_*/*` )
  #store length of paths array in a variable
  num_paths=${#paths[@]}
  
  path=${paths[$((($SUBTASK_ID-1) ))]}
  frame_name="$(cut -d'/' -f5 <<<"${path}")"
  
  echo "Running subtask $SUBTASK_ID"
  echo "frame index: $frame_name"

  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config "${path}/config.yaml" \
  --pretrained "${path}/model-600.pth" \
  --model-format full \
  --dataset-num-workers 6 \
  --log-dir "_results/logs/runs/best_models/best_delta_mask2former_idrej_inst_sem_conf"\
  --exp-name "${frame_name}"\
  --epochs 200\
  --inst-epoch-start 0\
  --sem-epoch-start 0\
  --voxel-raymarch-epoch-start 0\
  --inst-outlier-rejection \
  --inst-num-dilations 1\
  --prune-at-start \
  --inst-conf-enable \
  --sem-conf-enable \
  --max-depth 1.2\
  --inst-segment-reg-weight 0.1\
  --inst-segment-reg-epoch-start 101 \
  --load-modes imgs semantics instance preds_mask2former inst_conf sem_conf