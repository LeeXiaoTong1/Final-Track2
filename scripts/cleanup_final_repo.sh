#!/usr/bin/env bash
set -euo pipefail

# Clean Final-Track2 down to the stable robust three-branch reproduction path.
# Run locally from repository root, inspect `git status`, then commit.

rm -rf \
  .ipynb_checkpoints \
  backup_before_nextstep_branch_fusion \
  openbeats_src/.ipynb_checkpoints

rm -f \
  apply_soft_type_threshold.py \
  tune_soft_type_threshold.py \
  apply_calibrated_type_threshold.py \
  tune_calibrated_type_threshold.py \
  diagnose_type_posterior.py \
  apply_fixed_threshold.py \
  tune_branch_fusion.py \
  apply_branch_fusion.py \
  tune_multi_branch_fusion_holdout.py \
  apply_multi_branch_fusion.py \
  make_track2_type_labels.py \
  next_step_type_specialist_pipeline.sh \
  install_nextstep_branch_fusion.py \
  main_train_all95.py \
  main_train_all95_v2.py \
  dataset.py.bak_crop \
  model.py.bak_before_stable_cross_all95 \
  T2训练指令.txt \
  T2训练指令-随时更新.txt

# Keep final required scripts only:
#   main_train.py, model.py, dataset.py, config.py, generate_score_multicrop_plus.py,
#   train_type_classifier_track2.py, score_type_classifier_track2.py,
#   tune_three_branch_fusion_holdout.py, apply_three_branch_fusion.py,
#   patch_model_stable_cross_all95.py, docs/, submission_metadata/.

echo "Cleanup done. Review with: git status"
