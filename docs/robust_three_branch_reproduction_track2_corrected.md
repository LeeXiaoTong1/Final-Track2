# Robust Three-Branch Track2 Reproduction

This document describes the final stable Track2 system used by `zsx111` for the submitted evaluation result.

## Final system

The final stable submission is the **robust three-branch backoff system**:

1. `ft_xlsr_baseline`: FT-XLSR-AASIST / GDRO baseline branch.
2. `ufm_vocal_anchor_soundmusic`: UFM branch trained on full Track2 train/dev with Sound+Music-focused GDRO and vocal teacher anchoring.
3. `ufm_music_specialist`: all-type UFM Music-focused branch trained on full Track2 train/dev with Music-focused GDRO.
4. `type_classifier_xmb`: independent audio-type classifier.
5. `three_branch_fusion_eval_robust.json`: held-out Dev fusion parameters.

Important: this final stable system **does not** use type-filtered specialists trained with `label_by_type/train_speech.csv`, `label_by_type/train_music.csv`, or similar filtered label files. Those later type-filtered experiments generalized poorly on the full Evaluation set and are intentionally excluded from this reproduction path.

## Expected data layout

```text
AT_ADD_data/Track2/train
AT_ADD_data/Track2/dev
AT_ADD_data/Track2/eval
AT_ADD_data/Track2/label/train.csv
AT_ADD_data/Track2/label/dev.csv
```

## Expected pretrained models

```text
huggingface/wav2vec2-xls-r-300m
huggingface/MERT-v1-330M
huggingface/OpenBEATs-ICME
```

## Step 1. Prepare baseline branch

The final system used the following FT-XLSR-AASIST/GDRO checkpoint as the stable vocal branch and teacher:

```text
/root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/checkpoint/atadd_model_10.pt
```

Prepare a scoring directory:

```bash
mkdir -p ./ckpt_t2/ft_xlsr_baseline
cp /root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/args.json \
   ./ckpt_t2/ft_xlsr_baseline/args.json
cp /root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/checkpoint/atadd_model_10.pt \
   ./ckpt_t2/ft_xlsr_baseline/atadd_model.pt
```

## Step 2. Patch stable UFM cross block

```bash
python patch_model_stable_cross_all95.py
python -m py_compile model.py
```

## Step 3. Train independent audio-type classifier

```bash
python train_type_classifier_track2.py \
  --gpu 0 \
  --train_audio ./AT_ADD_data/Track2/train \
  --train_label ./AT_ADD_data/Track2/label/train.csv \
  --dev_audio ./AT_ADD_data/Track2/dev \
  --dev_label ./AT_ADD_data/Track2/label/dev.csv \
  --xlsr ./huggingface/wav2vec2-xls-r-300m \
  --mert ./huggingface/MERT-v1-330M \
  --beats ./huggingface/OpenBEATs-ICME \
  --out_dir ./ckpt_t2/type_classifier_xmb \
  --batch_size 32 \
  --epochs 5 \
  --lr 0.0001 \
  --seed 1234
```

Generate Dev/Eval type probabilities:

```bash
python score_type_classifier_track2.py \
  --gpu 0 \
  --model_dir ./ckpt_t2/type_classifier_xmb \
  --eval_audio ./AT_ADD_data/Track2/dev \
  --out_csv ./ckpt_t2/type_classifier_xmb/dev_type_probs.csv \
  --batch_size 32 \
  --num_workers 8

python score_type_classifier_track2.py \
  --gpu 0 \
  --model_dir ./ckpt_t2/type_classifier_xmb \
  --eval_audio ./AT_ADD_data/Track2/eval \
  --out_csv ./ckpt_t2/type_classifier_xmb/eval_type_probs.csv \
  --batch_size 32 \
  --num_workers 8
```

## Step 4. Train UFM vocal-anchor sound/music branch

This branch uses full Track2 train/dev labels and does not use type-filtered CSVs.

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
  --init_from ./ckpt_t2/ufm_all95_stage2_teacher_weak3/atadd_model.pt \
  --xlsr ./huggingface/wav2vec2-xls-r-300m \
  --mert ./huggingface/MERT-v1-330M \
  --beats ./huggingface/OpenBEATs-ICME \
  --ufm_freeze_xlsr \
  --ufm_freeze_mert \
  --ufm_freeze_beats \
  --ufm_dim 512 \
  --ufm_mem_slots 16 \
  --ufm_heads 8 \
  --ufm_layers 1 \
  --ufm_dropout 0.0 \
  --t2_return_type \
  --t2_gdro \
  --t2_gdro_active_types 1,3 \
  --t2_gdro_eta 0.35 \
  --ufm_type_loss 0.001 \
  --ufm_router_entropy 0.0 \
  --t2_teacher_model ft-w2v2aasist \
  --t2_sing_teacher_ckpt /root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/checkpoint/atadd_model_10.pt \
  --t2_teacher_anchor_types 0,2 \
  --t2_sing_anchor_weight 1.0 \
  --t2_sing_anchor_temp 2.0 \
  --t2_sing_anchor_margin_weight 0.2 \
  --t2_sing_anchor_correct_only \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 1 \
  --crop_consistency_weight 0.0 \
  --t2_target_floor 0.95 \
  --t2_floor_penalty 2.0 \
  --seed 1234 \
  --batch_size 32 \
  --lr 0.0000002 \
  --num_epochs 3 \
  --interval 3 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_vocal_anchor_soundmusic
```

## Step 5. Train all-type UFM music-focused branch

This is still an all-type branch. Do **not** replace the official full `train.csv` / `dev.csv` with `label_by_type/train_music.csv` or `label_by_type/dev_music.csv`.

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
  --init_from ./ckpt_t2/ufm_vocal_anchor_soundmusic/atadd_model.pt \
  --xlsr ./huggingface/wav2vec2-xls-r-300m \
  --mert ./huggingface/MERT-v1-330M \
  --beats ./huggingface/OpenBEATs-ICME \
  --ufm_freeze_xlsr \
  --ufm_freeze_mert \
  --ufm_freeze_beats \
  --ufm_dim 512 \
  --ufm_mem_slots 16 \
  --ufm_heads 8 \
  --ufm_layers 1 \
  --ufm_dropout 0.0 \
  --t2_return_type \
  --t2_gdro \
  --t2_gdro_active_types 3 \
  --t2_gdro_eta 0.60 \
  --ufm_type_loss 0.001 \
  --ufm_router_entropy 0.0 \
  --t2_teacher_model ft-w2v2aasist \
  --t2_sing_teacher_ckpt /root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/checkpoint/atadd_model_10.pt \
  --t2_teacher_anchor_types 2 \
  --t2_sing_anchor_weight 1.0 \
  --t2_sing_anchor_temp 2.0 \
  --t2_sing_anchor_margin_weight 0.2 \
  --t2_sing_anchor_correct_only \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 1 \
  --crop_consistency_weight 0.0 \
  --t2_target_floor 0.95 \
  --t2_floor_penalty 2.0 \
  --seed 1234 \
  --batch_size 32 \
  --lr 0.00000015 \
  --num_epochs 3 \
  --interval 3 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_music_specialist
```

## Step 6. Generate scores

Use `generate_score_multicrop_plus.py` for the baseline, UFM vocal-anchor branch, and all-type UFM music-focused branch. Use 5 deterministic crops and `mean_logit` aggregation.

Example for Dev baseline:

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ft_xlsr_baseline \
  --eval_task atadd-track2 \
  --eval_audio ./AT_ADD_data/Track2/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ft_xlsr_baseline/result/dev_baseline_plus.csv
```

Generate analogous files:

```text
./ckpt_t2/ufm_vocal_anchor_soundmusic/result/dev_ufm_plus.csv
./ckpt_t2/ufm_music_specialist/result/dev_music_plus.csv
./ckpt_t2/ft_xlsr_baseline/result/eval_baseline_plus.csv
./ckpt_t2/ufm_vocal_anchor_soundmusic/result/eval_ufm_plus.csv
./ckpt_t2/ufm_music_specialist/result/eval_music_plus.csv
```

## Step 7. Tune held-out three-branch fusion on Dev

```bash
python tune_three_branch_fusion_holdout.py \
  --baseline_csv ./ckpt_t2/ft_xlsr_baseline/result/dev_baseline_plus.csv \
  --ufm_csv ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/dev_ufm_plus.csv \
  --music_csv ./ckpt_t2/ufm_music_specialist/result/dev_music_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/dev_type_probs.csv \
  --label_csv ./AT_ADD_data/Track2/label/dev.csv \
  --out_json ./ckpt_t2/ufm_music_specialist/result/three_branch_fusion_eval_robust.json \
  --trials 8000 \
  --holdout_frac 0.50 \
  --mode music_boost \
  --seed 3407
```

## Step 8. Apply fusion to Eval

Do not tune on Eval.

```bash
python apply_three_branch_fusion.py \
  --baseline_csv ./ckpt_t2/ft_xlsr_baseline/result/eval_baseline_plus.csv \
  --ufm_csv ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/eval_ufm_plus.csv \
  --music_csv ./ckpt_t2/ufm_music_specialist/result/eval_music_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/eval_type_probs.csv \
  --calib_json ./ckpt_t2/ufm_music_specialist/result/three_branch_fusion_eval_robust.json \
  --out_csv ./ckpt_t2/predict_eval_three_branch_robust.csv \
  --debug_csv ./ckpt_t2/debug_eval_three_branch_robust.csv

cd ./ckpt_t2
cp predict_eval_three_branch_robust.csv predict.csv
zip submit_eval_three_branch_robust.zip predict.csv
```

## Final note

This reproduction intentionally excludes the later type-filtered specialist experiments. The stable final system is the three-branch robust backoff described above.
