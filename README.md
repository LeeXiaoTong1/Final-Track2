# Final-Track2: AT-ADD 2026 Track 2 Robust Three-Branch System

This repository contains the reproducible code path for the final Track 2 submission by `zsx111`.

Final submitted system: **robust three-branch backoff fusion**.

The final system intentionally keeps only the components needed for the stable final submission:

1. **FT-XLSR-AASIST baseline branch** for stable vocal-class behavior.
2. **UFM vocal-anchor sound/music branch** trained on full Track2 train/dev labels with Sound+Music-focused GDRO and vocal teacher anchoring.
3. **All-type UFM music-focused branch** trained on full Track2 train/dev labels with Music-focused GDRO.
4. **Independent audio-type classifier** for Speech/Sound/Singing/Music routing.
5. **Held-out three-branch fusion** for final prediction.

Important: the final stable system does **not** use the later type-filtered specialists (`label_by_type/train_speech.csv`, `label_by_type/train_music.csv`, etc.). Those experiments were found to generalize poorly on the full Evaluation set and are intentionally excluded from the final reproduction path.

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

## Quick reproduction

See:

```text
docs/robust_three_branch_reproduction_track2_corrected.md
```

The expected data layout is:

```text
AT_ADD_data/Track2/train
AT_ADD_data/Track2/dev
AT_ADD_data/Track2/eval
AT_ADD_data/Track2/label/train.csv
AT_ADD_data/Track2/label/dev.csv
```

The expected local pretrained checkpoints are:

```text
huggingface/wav2vec2-xls-r-300m
huggingface/MERT-v1-330M
huggingface/OpenBEATs-ICME
```

## Cleanup policy

Files under `.ipynb_checkpoints/`, `backup_before_nextstep_branch_fusion/`, old patch scripts, type-filtered specialist scripts, and debug/tuning scripts not used by the final robust-backoff system should not be used for the final submission.

A local cleanup helper is provided:

```bash
bash scripts/cleanup_final_repo.sh
```

Run it only after cloning locally if you want to remove stale experiment files from your working tree.
