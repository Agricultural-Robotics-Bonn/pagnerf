#block(name=grid_tests,threads=8,gpus=1,memory=10000,hours=6, subtasks=280)
  paths=( `echo _results/logs/runs/best_delta_deeplab_depth_filtered/test_frame_*/*` )

  # lods=( "21" "18" "15" "12" "9")
  lods=( "7" "5" "3")
  num_lods_exps=${#lods[@]}
  #store length of paths array in a variable
  num_paths=${#paths[@]}
  echo "num paths: $num_paths"

  INDEX=$((SUBTASK_ID-1))
  LOD_IDX=$((INDEX % num_lods_exps))
  PATH_IDX=$((INDEX / num_lods_exps))

  num_lods=${lods[$LOD_IDX]}
  path=${paths[$PATH_IDX]}
  frame_name="$(cut -d'/' -f5 <<<"${path}")"
  
  echo "index $INDEX"
  echo "Running subtask $SUBTASK_ID"
  echo "Frame name: $frame_name"
  echo "Num LODs: $num_lods"

  finest_scale=0.001

  base_path="_results/logs/runs/best_models/lod_ablation_fixed"
  exp_name="fscale_${finest_scale}_lods_${num_lods}/${frame_name}"
  exp_path="${base_path}/${exp_name}"

  if [ -d "${exp_path}" ] 
  then
      echo "LODSKIPPING Experiment already run, skipping..."
      exit 1
  fi

  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config "${path}/config.yaml" \
  --dataset-num-workers 6 \
  --log-dir "${base_path}"\
  --exp-name "${exp_name}"\
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