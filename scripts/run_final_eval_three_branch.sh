#!/usr/bin/env bash
set -euo pipefail

# Final Track2 robust three-branch Eval inference.
# Run from repository root after all checkpoints and Dev-tuned fusion JSON are prepared.

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

EVAL_AUDIO=${EVAL_AUDIO:-./AT_ADD_data/Track2/eval}
FUSION_JSON=${FUSION_JSON:-./ckpt_t2/ufm_music_specialist/result/three_branch_fusion_eval_robust.json}

python score_type_classifier_track2.py \
  --gpu 0 \
  --model_dir ./ckpt_t2/type_classifier_xmb \
  --eval_audio "$EVAL_AUDIO" \
  --out_csv ./ckpt_t2/type_classifier_xmb/eval_type_probs.csv \
  --batch_size 32 \
  --num_workers 8

python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ft_xlsr_baseline \
  --eval_task atadd-track2 \
  --eval_audio "$EVAL_AUDIO" \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ft_xlsr_baseline/result/eval_baseline_plus.csv

python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_vocal_anchor_soundmusic \
  --eval_task atadd-track2 \
  --eval_audio "$EVAL_AUDIO" \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/eval_ufm_plus.csv

python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_music_specialist \
  --eval_task atadd-track2 \
  --eval_audio "$EVAL_AUDIO" \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_music_specialist/result/eval_music_plus.csv

python apply_three_branch_fusion.py \
  --baseline_csv ./ckpt_t2/ft_xlsr_baseline/result/eval_baseline_plus.csv \
  --ufm_csv ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/eval_ufm_plus.csv \
  --music_csv ./ckpt_t2/ufm_music_specialist/result/eval_music_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/eval_type_probs.csv \
  --calib_json "$FUSION_JSON" \
  --out_csv ./ckpt_t2/predict_eval_three_branch_robust.csv \
  --debug_csv ./ckpt_t2/debug_eval_three_branch_robust.csv

cd ./ckpt_t2
cp predict_eval_three_branch_robust.csv predict.csv
zip -f submit_eval_three_branch_robust.zip predict.csv || zip submit_eval_three_branch_robust.zip predict.csv

echo "Done: ckpt_t2/submit_eval_three_branch_robust.zip"
