#block(name=abl_sep_7_4,threads=8,gpus=1,memory=10000,hours=6, subtasks=360)
  # capacities=( "18" "16" "14" "12" "10" "8")
  # capacities=( "18" "17" "16" "15" "14" "13" "12" "11" "10" "9" "8")
  capacities=( "7" "6" "5" "4")
  # capacities=( "12" "14" "16" "18")
  # capacities=( "8" "9" "10" "11")
  num_caps=${#capacities[@]}

  paths=( `echo _results/logs/runs/best_delta_deeplab_depth_filtered/test_frame_*/*` )

  #store length of paths array in a variable
  num_paths=${#paths[@]}
  
  capacity=${capacities[$((($SUBTASK_ID-1) % $num_caps))]}
  path=${paths[$((($SUBTASK_ID-1) / num_caps))]}
  frame_name="$(cut -d'/' -f5 <<<"${path}")"
  
  echo "Running subtask $SUBTASK_ID"
  echo "model log2 capacity: $capacity"
  echo "frame index: $frame_name"

  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config "${path}/config.yaml" \
  --pretrained "${path}/model-600.pth" \
  --model-format params_only_ignore_missmatch\
  --log-dir "_results/logs/runs/best_models/bup20_capacity_ablation_separate"\
  --exp-name "delta_cap_${capacity}/${frame_name}"\
  --delta-capacity-log-2 "${capacity}"\
  --panoptic-features-type separate\
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

