#block(name=color_abl_15_6,threads=8,gpus=1,memory=10000,hours=6, subtasks=360)
  capacities=( "15" "12" "9" "6")
  # capacities=( "18" "17" "16" "15" "14" "13" "12" "11" "10" "9" "8")
  # capacities=( "7" "6" "5" "4")
  # capacities=( "12" "14" "16" "18")
  # capacities=( "8" "9" "10" "11")
  num_caps=${#capacities[@]}

  paths=( `echo _results/logs/runs/best_models/lod_ablation/fscale_0.001_lods_9/test_frame_*/*` )

  #store length of paths array in a variable
  num_paths=${#paths[@]}
  
  capacity=${capacities[$((($SUBTASK_ID-1) % $num_caps))]}
  path=${paths[$((($SUBTASK_ID-1) / num_caps))]}
  frame_name="$(cut -d'/' -f7 <<<"${path}")"
  
  echo "Running subtask $SUBTASK_ID"
  echo "model log2 capacity: $capacity"
  echo "frame index: $frame_name"
  echo "color_cap_${capacity}/${frame_name}"\

  base_path="_results/logs/runs/best_models/color_ablation_fixed/"
  exp_name="color_cap_${capacity}/${frame_name}"
  exp_path="${base_path}/${exp_name}"

  if [ -d "${exp_path}" ] 
  then
      echo "Experiment already run, skipping..."
      exit 1
  fi

  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config "${path}/config.yaml" \
  --model-format params_only_ignore_missmatch\
  --log-dir "${base_path}"\
  --exp-name "${exp_name}"\
  --capacity-log-2 "${capacity}"\
  --num-lods 9\
  --delta-capacity-log-2 18\