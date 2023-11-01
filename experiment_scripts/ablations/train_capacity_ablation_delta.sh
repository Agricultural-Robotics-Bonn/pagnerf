#block(name=abl_panop_async2,threads=8,gpus=1,memory=10000,hours=6, subtasks=450)
  # capacities=( "18" "16" "14" "12" "10" "8")
  # capacities=( "18" "17" "16" "15" "14" "13" "12" "11" "10" "9" "8")
  capacities=( "15" "12" "9" "6" "3")
  # capacities=( "15" "12")
  # capacities=( "9" "6")
  # capacities=( "12" "14" "16" "18")
  # capacities=( "8" "9" "10" "11")
  num_caps=${#capacities[@]}

  paths=( `echo _results/logs/runs/best_models/color_ablation_fixed/color_cap_15/test_frame_*/*` )

  #store length of paths array in a variable
  num_paths=${#paths[@]}
  
  capacity=${capacities[$((($SUBTASK_ID-1) % $num_caps))]}
  path=${paths[$((($SUBTASK_ID-1) / num_caps))]}
  frame_name="$(cut -d'/' -f7 <<<"${path}")"
  
  echo "Running subtask $SUBTASK_ID"
  echo "model log2 capacity: $capacity"
  echo "frame index: $frame_name"

  base_path="_results/logs/runs/best_models/panoptic_ablation_delta_fixed/"
  exp_name="panoptic_cap_${capacity}/${frame_name}"
  exp_path="${base_path}/${exp_name}"

  if [ -d "${exp_path}" ] 
  then
      echo "Experiment already run, skipping..."
      exit 1
  fi

  mkdir -p $exp_path

  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config "${path}/config.yaml" \
  --log-dir "${base_path}"\
  --exp-name "${exp_name}"\
  --delta-capacity-log-2 "${capacity}"\
  --panoptic-features-type delta