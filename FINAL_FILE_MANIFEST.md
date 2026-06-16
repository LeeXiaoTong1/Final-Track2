# Final File Manifest

This repository is intended to preserve only the stable Track2 robust-backoff reproduction path.

## Required core code

- `main_train.py`
- `model.py`
- `dataset.py`
- `config.py`
- `feature_extraction.py`
- `exp/feature_extraction_exp.py`
- `eval_metrics.py`
- `utils.py`
- `backbone/rawaasist.py`
- `openbeats_src/`

## Required final scripts

- `patch_model_stable_cross_all95.py`
- `train_type_classifier_track2.py`
- `score_type_classifier_track2.py`
- `generate_score_multicrop_plus.py`
- `tune_three_branch_fusion_holdout.py`
- `apply_three_branch_fusion.py`
- `scripts/run_final_eval_three_branch.sh`
- `scripts/cleanup_final_repo.sh`

## Required documentation / metadata

- `README.md`
- `docs/robust_three_branch_reproduction_track2_corrected.md`
- `submission_metadata/zsx1111_track2_90.77_corrected.yaml`

## Explicitly excluded from final reproduction

The following are not part of the final stable system and should be removed from a clean local copy:

- `.ipynb_checkpoints/`
- `backup_before_nextstep_branch_fusion/`
- old patch scripts such as `patch_main_train_*`, `repair_main_train_*`, `install_nextstep_branch_fusion.py`
- soft-threshold calibration scripts not used by final reproduction
- multi-branch or type-filtered specialist scripts not used by final reproduction
- type-filtered label generation scripts and `label_by_type` training route

The final stable system is **not** the later type-filtered specialist route. It is:

```text
baseline + UFM vocal-anchor sound/music branch + all-type UFM music-focused branch + independent type classifier + held-out three-branch fusion
```
