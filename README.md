# Final-Track2: AT-ADD 2026 Track 2 Robust Three-Branch System

This repository contains the reproducible code path for the final AT-ADD 2026 Track 2 submission by `zsx111`.

Final submitted system: **robust three-branch backoff fusion**.

The final system intentionally keeps only the components needed for the stable full-Evaluation submission:

1. **FT-XLSR-AASIST baseline branch** for stable vocal-class behavior.
2. **UFM vocal-anchor Sound/Music branch** trained on full Track2 train/dev labels with Sound+Music-focused GDRO and vocal teacher anchoring.
3. **All-type UFM Music-focused branch** trained on full Track2 train/dev labels with Music-focused GDRO.
4. **Independent audio-type classifier** for Speech/Sound/Singing/Music routing.
5. **Held-out three-branch fusion** for final prediction.

Important: the final stable system **does not** use the later type-filtered specialists (`label_by_type/train_speech.csv`, `label_by_type/train_music.csv`, etc.). Those experiments were found to generalize poorly on the full Evaluation set and are intentionally excluded from the final reproduction path.

---

## Final result

Best full Evaluation result:

| Metric | Score |
|---|---:|
| Macro-F1 | 90.77 |
| Speech | 81.88 |
| Sound | 92.55 |
| Singing | 95.47 |
| Music | 93.15 |

---

## Core files

Training / model code:

- `main_train.py`
- `model.py`
- `dataset.py`
- `config.py`
- `feature_extraction.py`
- `exp/feature_extraction_exp.py`
- `backbone/rawaasist.py`
- `openbeats_src/`

Final reproduction scripts:

- `patch_model_stable_cross_all95.py`
- `train_type_classifier_track2.py`
- `score_type_classifier_track2.py`
- `generate_score_multicrop_plus.py`
- `tune_three_branch_fusion_holdout.py`
- `apply_three_branch_fusion.py`
- `docs/robust_three_branch_reproduction_track2_corrected.md`
- `submission_metadata/zsx1111_track2_90.77_corrected.yaml`

---

## Expected data layout

Place official Track2 data as follows:

```text
AT_ADD_data/Track2/train
AT_ADD_data/Track2/dev
AT_ADD_data/Track2/eval
AT_ADD_data/Track2/label/train.csv
AT_ADD_data/Track2/label/dev.csv
```

Do not use Evaluation labels. Evaluation is only used for score generation and fixed fusion application.

---

## Expected local pretrained checkpoints

The repository expects the following local pretrained frontends:

```text
huggingface/wav2vec2-xls-r-300m
huggingface/MERT-v1-330M
huggingface/OpenBEATs-ICME
```

The final system also uses an internally trained FT-XLSR-AASIST/GDRO baseline checkpoint as the baseline score branch and teacher-anchor checkpoint. In our experiments, this checkpoint was:

```text
/root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/checkpoint/atadd_model_10.pt
```

This checkpoint is **not** an external pretrained model. It is an internally trained Track2 baseline/teacher checkpoint trained using official Track2 train/dev data. For reproduction, it can be replaced by an equivalent dev-selected FT-XLSR-AASIST baseline checkpoint trained only with official Track2 train/dev data.

---

# Full reproduction commands

All commands below are written relative to the repository root.

```bash
cd /path/to/Final-Track2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## Step 1. Prepare baseline branch

Create a scoring directory for the FT-XLSR-AASIST/GDRO baseline branch:

```bash
mkdir -p ./ckpt_t2/ft_xlsr_baseline

cp /root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/args.json \
   ./ckpt_t2/ft_xlsr_baseline/args.json

cp /root/autodl-tmp/AT-ADD-Baseline-track2/ckpt_t2/gdro_adv_xlsr_aasist/checkpoint/atadd_model_10.pt \
   ./ckpt_t2/ft_xlsr_baseline/atadd_model.pt
```

If you use another equivalent FT-XLSR-AASIST baseline checkpoint, keep the same folder structure:

```text
ckpt_t2/ft_xlsr_baseline/args.json
ckpt_t2/ft_xlsr_baseline/atadd_model.pt
```

---

## Step 2. Patch stable UFM cross block

The original cross-stream block based on bidirectional MultiheadAttention was unstable in our `--ufm_layers 1` training. The final system uses a stable gated cross-stream block.

```bash
python patch_model_stable_cross_all95.py
python -m py_compile model.py
```

---

## Step 3. Train independent audio-type classifier

The independent audio-type classifier predicts Speech/Sound/Singing/Music posterior probabilities used by the final fusion module. It is separate from the UFM detector's internal type posterior.

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

Generate Dev type probabilities:

```bash
python score_type_classifier_track2.py \
  --gpu 0 \
  --model_dir ./ckpt_t2/type_classifier_xmb \
  --eval_audio ./AT_ADD_data/Track2/dev \
  --out_csv ./ckpt_t2/type_classifier_xmb/dev_type_probs.csv \
  --batch_size 32 \
  --num_workers 8
```

Generate Eval type probabilities:

```bash
python score_type_classifier_track2.py \
  --gpu 0 \
  --model_dir ./ckpt_t2/type_classifier_xmb \
  --eval_audio ./AT_ADD_data/Track2/eval \
  --out_csv ./ckpt_t2/type_classifier_xmb/eval_type_probs.csv \
  --batch_size 32 \
  --num_workers 8
```

---

## Step 4. Train UFM vocal-anchor Sound/Music branch

This branch uses the **full Track2 train/dev labels**. It does **not** use type-filtered CSVs.

Main role:

- improve Sound/Music robustness;
- keep vocal classes stable through teacher anchoring;
- provide the main UFM score branch for final fusion.

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
  --t2_sing_teacher_ckpt ./ckpt_t2/ft_xlsr_baseline/atadd_model.pt \
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

If the warm-start checkpoint `./ckpt_t2/ufm_all95_stage2_teacher_weak3/atadd_model.pt` is unavailable, train the earlier UFM stage first or replace it with the closest dev-selected UFM checkpoint trained only on official Track2 train/dev data.

---

## Step 5. Train all-type UFM Music-focused branch

This branch is still trained with the **full Track2 train/dev distribution**. Do **not** replace the official full `train.csv` / `dev.csv` with `label_by_type/train_music.csv` or `label_by_type/dev_music.csv`.

Main role:

- improve Music robustness;
- avoid overfitting caused by music-only type-filtered training;
- provide a specialized but all-type-constrained score branch for final fusion.

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
  --t2_sing_teacher_ckpt ./ckpt_t2/ft_xlsr_baseline/atadd_model.pt \
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

---

## Step 6. Generate Dev scores for fusion tuning

Use `generate_score_multicrop_plus.py` with 5 deterministic crops and `mean_logit` aggregation.

### Baseline Dev score

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

### UFM vocal-anchor Dev score

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_vocal_anchor_soundmusic \
  --eval_task atadd-track2 \
  --eval_audio ./AT_ADD_data/Track2/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/dev_ufm_plus.csv
```

### UFM Music-focused Dev score

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_music_specialist \
  --eval_task atadd-track2 \
  --eval_audio ./AT_ADD_data/Track2/dev \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_music_specialist/result/dev_music_plus.csv
```

---

## Step 7. Tune held-out three-branch fusion on Dev

Fusion is tuned only on Dev. Evaluation data are never used for threshold or fusion fitting.

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

The resulting JSON stores the final fusion parameters used by the robust fallback system.

---

## Step 8. Generate Eval scores

### Baseline Eval score

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ft_xlsr_baseline \
  --eval_task atadd-track2 \
  --eval_audio ./AT_ADD_data/Track2/eval \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ft_xlsr_baseline/result/eval_baseline_plus.csv
```

### UFM vocal-anchor Eval score

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_vocal_anchor_soundmusic \
  --eval_task atadd-track2 \
  --eval_audio ./AT_ADD_data/Track2/eval \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/eval_ufm_plus.csv
```

### UFM Music-focused Eval score

```bash
python generate_score_multicrop_plus.py \
  --gpu 0 \
  --model_path ./ckpt_t2/ufm_music_specialist \
  --eval_task atadd-track2 \
  --eval_audio ./AT_ADD_data/Track2/eval \
  --num_crops 5 \
  --batch_files 8 \
  --num_workers 8 \
  --agg mean_logit \
  --score_file ./ckpt_t2/ufm_music_specialist/result/eval_music_plus.csv
```

---

## Step 9. Apply fixed Dev-tuned fusion to Eval

Do not tune on Eval. Only apply the fusion JSON learned from Dev.

```bash
python apply_three_branch_fusion.py \
  --baseline_csv ./ckpt_t2/ft_xlsr_baseline/result/eval_baseline_plus.csv \
  --ufm_csv ./ckpt_t2/ufm_vocal_anchor_soundmusic/result/eval_ufm_plus.csv \
  --music_csv ./ckpt_t2/ufm_music_specialist/result/eval_music_plus.csv \
  --type_csv ./ckpt_t2/type_classifier_xmb/eval_type_probs.csv \
  --calib_json ./ckpt_t2/ufm_music_specialist/result/three_branch_fusion_eval_robust.json \
  --out_csv ./ckpt_t2/predict_eval_three_branch_robust.csv \
  --debug_csv ./ckpt_t2/debug_eval_three_branch_robust.csv
```

---

## Step 10. Package final submission

```bash
cd ./ckpt_t2
cp predict_eval_three_branch_robust.csv predict.csv
zip submit_eval_three_branch_robust.zip predict.csv
```

Submit:

```text
ckpt_t2/submit_eval_three_branch_robust.zip
```

---

## What is intentionally excluded

The final stable system intentionally excludes the following later experiments:

- raw UFM type-posterior dynamic thresholding;
- soft/calibrated threshold scripts;
- type-filtered specialists trained with `label_by_type/train_*.csv`;
- speech-only / sound-only / music-only specialist branches;
- over-complex multi-branch specialist fusion.

These experiments were useful during development but were not part of the final robust Evaluation submission because they showed weaker full-Eval generalization.

---

## Cleanup policy

Files under `.ipynb_checkpoints/`, `backup_before_nextstep_branch_fusion/`, old patch scripts, type-filtered specialist scripts, and debug/tuning scripts not used by the final robust-backoff system should not be used for the final submission.

If present, remove them before final release or keep them only in a separate archive branch.
