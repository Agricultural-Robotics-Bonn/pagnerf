#block(name=panoli_conf,threads=8,gpus=1,memory=10000,hours=6, subtasks=100)
  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WISP_HEADLESS=1 \
  python main_interactive.py \
  --config configs/bup20/panoptic_lifting_app.yaml \
  --dataset-num-workers 6 \
  --dataset-center-idx "${SUBTASK_ID-1}"\
  --log-dir "_results/logs/runs/best_models/bup20_panoli_conf"\
  --exp-name "test_frame_${SUBTASK_ID-1}"\
  --inst-conf-enable \
  --sem-conf-enable \
  --load-modes imgs semantics instance preds_mask2former inst_conf sem_conf

  