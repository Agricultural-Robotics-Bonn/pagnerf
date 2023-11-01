#block(name=fix_lods_debug,threads=8,gpus=1,memory=10000,hours=6, subtasks=460)
  paths=( `echo _results/logs/runs/best_models/lod_ablation/*/test_frame_*/*` )

  #store length of paths array in a variable
  num_paths=${#paths[@]}
  echo "num paths: $num_paths"

  INDEX=$((SUBTASK_ID-1))

  path=${paths[$INDEX]}
  ablation_name="$(cut -d'/' -f6 <<<"${path}")"
  frame_name="$(cut -d'/' -f7 <<<"${path}")"
  timestamp="$(cut -d'/' -f8 <<<"${path}")"
  
  echo "index $INDEX"
  echo "Running subtask $SUBTASK_ID"
  echo "Frame name: $frame_name"
  echo "Ablation name: $ablation_name"

  fixed_path="_results/logs/runs/best_models/lod_ablation_fixed/${ablation_name}"

  checkpoints=( `echo ${path}/model-*.pth` )
  last_model=$(basename -- "${checkpoints[-1]}")

  if [ "$last_model" == "model-800.pth" ]
  then
    echo "valid model, copying resutls"

    mkdir -p $fixed_path/$frame_name/$timestamp #mkdir
    cp $path/config.yaml $fixed_path/$frame_name/$timestamp/config.yaml #cp
    cp $path/log.txt $fixed_path/$frame_name/$timestamp/log.txt         #cp
    cp $path/metrics.csv $fixed_path/$frame_name/$timestamp/metrics.csv #cp
    exit 1
  fi

  echo "Final result missing, training from checkpoint ${last_model}"


  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config "${path}/config.yaml" \
  --pretrained "${checkpoints[-1]}"\
  --dataset-num-workers 6 \
  --log-dir "${fixed_path}"\
  --exp-name "${frame_name}"\
  --epochs 200\
  --inst-segment-reg-weight 0.1\
  --inst-segment-reg-epoch-start 101 \
  --sem-epoch-start 0\
  --voxel-raymarch-epoch-start 0\
  --inst-epoch-start 0\