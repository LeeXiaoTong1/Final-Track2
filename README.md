# Final-Track2：AT-ADD 2026 Track 2 稳健三分支回退系统

本仓库用于复现 `zsx111` 在 AT-ADD 2026 Track 2 中最终提交的**稳健回退版**系统。

最终提交系统不是单一模型，而是：

```text
FT-XLSR-AASIST baseline branch
+ UFM vocal-anchor Sound/Music branch
+ all-type UFM Music-focused branch
+ independent audio-type classifier
+ held-out Dev three-branch fusion
```

最终 full Evaluation 结果：

| Metric | Score |
|---|---:|
| Macro-F1 | 90.77 |
| Speech | 81.88 |
| Sound | 92.55 |
| Singing | 95.47 |
| Music | 93.15 |

---

## 1. 最终系统组成

最终稳健回退版包含五个部分：

1. **FT-XLSR-AASIST baseline branch**：提供稳定 vocal 类行为，并作为 teacher-anchor 来源。
2. **UFM vocal-anchor Sound/Music branch**：使用完整 Track2 train/dev 标签，通过 Sound+Music-focused GDRO 强化 Sound/Music，同时使用 teacher anchor 保护 vocal 类。
3. **All-type UFM Music-focused branch**：仍然使用完整 Track2 train/dev 标签，只通过 Music-focused GDRO 强化 Music，不使用 music-only 过滤数据。
4. **Independent audio-type classifier**：独立预测 Speech/Sound/Singing/Music 类型概率，用于最终融合路由。
5. **Held-out three-branch fusion**：只在 Dev held-out split 上学习融合参数；Eval 阶段只加载固定 fusion JSON。

重要：最终系统**不使用** `label_by_type/train_speech.csv`、`label_by_type/train_music.csv` 等 type-filtered specialists。相关实验在 Dev 或 Progress 上可能有短期增益，但 full Evaluation 泛化较弱，因此不属于最终复现路径。

---

## 2. 核心文件

训练与模型代码：

- `main_train.py`
- `model.py`
- `dataset.py`
- `config.py`
- `feature_extraction.py`
- `exp/feature_extraction_exp.py`
- `backbone/rawaasist.py`
- `openbeats_src/`

最终复现脚本：

- `patch_model_stable_cross_all95.py`
- `train_type_classifier_track2.py`
- `score_type_classifier_track2.py`
- `generate_score_multicrop_plus.py`
- `tune_three_branch_fusion_holdout.py`
- `apply_three_branch_fusion.py`

---

## 3. 数据与预训练模型目录

官方 Track2 数据目录：

```text
AT_ADD_data/Track2/train
AT_ADD_data/Track2/dev
AT_ADD_data/Track2/eval
AT_ADD_data/Track2/label/train.csv
AT_ADD_data/Track2/label/dev.csv
```

本地预训练模型目录：

```text
huggingface/wav2vec2-xls-r-300m
huggingface/MERT-v1-330M
huggingface/OpenBEATs-ICME
```

说明：

- 训练、checkpoint 选择、阈值搜索、融合调参只使用官方 train/dev。
- Eval 只用于生成分数、生成类型概率、应用 Dev 上学到的固定融合参数和打包提交。
- 不使用 Eval 标签，不在 Eval 上调阈值，不在 Eval 上调融合。

---

# 4. 从空目录完整复现所有 checkpoint

所有命令默认在仓库根目录执行：

```bash
cd /path/to/Final-Track2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p ./ckpt_t2
```

为了避免覆盖旧实验，建议在一个新的工作目录或新的 `ckpt_t2` 输出目录中运行以下命令。

---

## Step 0：修补稳定 UFM cross block

最终稳健版使用 stable gated cross-stream block，避免原始双向 MultiheadAttention cross block 在 `--ufm_layers 1` 下出现数值不稳定。

```bash
python patch_model_stable_cross_all95.py

python -m py_compile \
  main_train.py model.py dataset.py config.py \
  generate_score_multicrop_plus.py \
  train_type_classifier_track2.py \
  score_type_classifier_track2.py \
  tune_three_branch_fusion_holdout.py \
  apply_three_branch_fusion.py
```

---

## Step 1：从零训练 FT-XLSR-AASIST baseline branch

该 checkpoint 后续同时作为：

- 最终融合中的 baseline score branch；
- UFM teacher-anchor 的 teacher checkpoint。

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ft-w2v2aasist \
  --xlsr ./huggingface/wav2vec2-xls-r-300m \
  --t2_return_type \
  --t2_gdro \
  --t2_gdro_eta 0.35 \
  --t2_type_adv \
  --t2_type_adv_weight 0.05 \
  --t2_grl_lambda 1.0 \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 1 \
  --crop_consistency_weight 0.0 \
  --t2_singing_floor 0.94 \
  --t2_singing_penalty 1.5 \
  --seed 1234 \
  --batch_size 32 \
  --num_workers 8 \
  --lr 0.00001 \
  --num_epochs 10 \
  --interval 5 \
  --save_best_by safe_f1 \
  --out_fold ./ckpt_t2/ft_xlsr_baseline
```

训练完成后应存在：

```text
ckpt_t2/ft_xlsr_baseline/args.json
ckpt_t2/ft_xlsr_baseline/atadd_model.pt
```

后续默认使用：

```text
ckpt_t2/ft_xlsr_baseline/atadd_model.pt
```

---

## Step 2：从零训练 UFM all-type seed checkpoint

该阶段生成 UFM 初始 checkpoint。它使用完整 Track2 train/dev 分布，不使用 type-filtered CSV。

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
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
  --t2_gdro_active_types 0,1,2,3 \
  --t2_gdro_eta 0.35 \
  --ufm_type_loss 0.001 \
  --ufm_router_entropy 0.0 \
  --train_crop_mode random \
  --dev_crop_mode head \
  --train_num_crops 1 \
  --crop_consistency_weight 0.0 \
  --t2_target_floor 0.95 \
  --t2_floor_penalty 2.0 \
  --seed 1234 \
  --batch_size 32 \
  --num_workers 8 \
  --lr 0.0001 \
  --num_epochs 3 \
  --interval 3 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_all95_seed
```

输出 checkpoint：

```text
ckpt_t2/ufm_all95_seed/atadd_model.pt
```

---

## Step 3：训练 UFM all95 stage2 teacher-weak3 checkpoint

该阶段从 UFM seed 继续训练，使用 teacher anchor 保护 Singing，并将 GDRO 聚焦到 Speech/Sound/Music 三个 weak types。该 checkpoint 是最终 UFM vocal-anchor Sound/Music branch 的 warm-start 来源。

```bash
python main_train.py \
  --gpu 0 \
  --train_task atadd-track2 \
  --model ufm-track2-full \
  --init_from ./ckpt_t2/ufm_all95_seed/atadd_model.pt \
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
  --t2_gdro_active_types 0,1,3 \
  --t2_gdro_eta 0.35 \
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
  --num_workers 8 \
  --lr 0.0000002 \
  --num_epochs 3 \
  --interval 3 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_all95_stage2_teacher_weak3
```

输出 checkpoint：

```text
ckpt_t2/ufm_all95_stage2_teacher_weak3/atadd_model.pt
```

---

## Step 4：训练独立 audio-type classifier

该模型只预测音频类型，不预测 real/fake。输出的类型概率用于最终三分支融合。

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

生成 Dev 类型概率：

```bash
python score_type_classifier_track2.py \
  --gpu 0 \
  --model_dir ./ckpt_t2/type_classifier_xmb \
  --eval_audio ./AT_ADD_data/Track2/dev \
  --out_csv ./ckpt_t2/type_classifier_xmb/dev_type_probs.csv \
  --batch_size 32 \
  --num_workers 8
```

生成 Eval 类型概率：

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

## Step 5：训练 UFM vocal-anchor Sound/Music branch

该分支是最终三分支融合中的第二个 score branch，输出 `s_u(x)`。

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
  --num_workers 8 \
  --lr 0.0000002 \
  --num_epochs 3 \
  --interval 3 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_vocal_anchor_soundmusic
```

输出 checkpoint：

```text
ckpt_t2/ufm_vocal_anchor_soundmusic/atadd_model.pt
```

---

## Step 6：训练 all-type UFM Music-focused branch

该分支是最终三分支融合中的第三个 score branch，输出 `s_m(x)`。它仍然使用完整 Track2 train/dev 分布，只把 GDRO active type 设置为 Music。

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
  --num_workers 8 \
  --lr 0.00000015 \
  --num_epochs 3 \
  --interval 3 \
  --save_best_by all95_f1 \
  --out_fold ./ckpt_t2/ufm_music_specialist
```

输出 checkpoint：

```text
ckpt_t2/ufm_music_specialist/atadd_model.pt
```

---

# 5. Dev 分数生成与融合调参

## Step 7：生成三个分支的 Dev score

使用 5 deterministic crops 和 `mean_logit` 聚合。

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

## Step 8：在 Dev 上学习 held-out three-branch fusion

只在 Dev 上进行 fusion tuning。Eval 不参与任何阈值或融合参数学习。

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

输出融合参数：

```text
ckpt_t2/ufm_music_specialist/result/three_branch_fusion_eval_robust.json
```

---

# 6. Eval 推理与提交打包

## Step 9：生成三个分支的 Eval score

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

## Step 10：应用 Dev 上学习到的固定 fusion

不要在 Eval 上重新调参。只加载 Dev 上得到的 fusion JSON。

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

## Step 11：打包最终提交文件

```bash
cd ./ckpt_t2
cp predict_eval_three_branch_robust.csv predict.csv
zip submit_eval_three_branch_robust.zip predict.csv
```

最终提交文件：

```text
ckpt_t2/submit_eval_three_branch_robust.zip
```

---

# 7. 快速检查清单

完成训练和推理后，建议检查：

```bash
ls ./ckpt_t2/ft_xlsr_baseline/atadd_model.pt
ls ./ckpt_t2/ufm_all95_seed/atadd_model.pt
ls ./ckpt_t2/ufm_all95_stage2_teacher_weak3/atadd_model.pt
ls ./ckpt_t2/type_classifier_xmb/dev_type_probs.csv
ls ./ckpt_t2/type_classifier_xmb/eval_type_probs.csv
ls ./ckpt_t2/ufm_vocal_anchor_soundmusic/atadd_model.pt
ls ./ckpt_t2/ufm_music_specialist/atadd_model.pt
ls ./ckpt_t2/ufm_music_specialist/result/three_branch_fusion_eval_robust.json
ls ./ckpt_t2/predict_eval_three_branch_robust.csv
ls ./ckpt_t2/submit_eval_three_branch_robust.zip
```

---

# 8. 最终系统明确排除的内容

最终稳健回退版不包含以下路线：

- raw UFM type-posterior dynamic thresholding；
- soft threshold / calibrated threshold；
- `label_by_type/train_*.csv` type-filtered specialists；
- speech-only / sound-only / music-only specialist branches；
- 过复杂 multi-branch specialist fusion；
- 使用 Progress/Eval 标签进行训练、调阈值或调融合。

这些实验可以作为技术报告中的失败经验，但不属于最终提交系统。

---

# 9. 复现说明

由于深度学习训练存在随机性，不同 GPU、CUDA/cuDNN、PyTorch 版本和数据读取顺序可能导致分数有小幅波动。本 README 的目标是给出**从空目录训练出最终稳健回退版所有 checkpoint 的完整路径**，而不是保证 bit-wise identical 的 checkpoint。

最终稳健系统结构必须保持为：

```text
FT-XLSR-AASIST baseline branch
+ UFM vocal-anchor Sound/Music branch
+ all-type UFM Music-focused branch
+ independent audio-type classifier
+ held-out Dev three-branch fusion
```
