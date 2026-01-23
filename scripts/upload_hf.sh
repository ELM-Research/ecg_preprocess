# uv run src/upload_hf.py \
# --map_json src/ecg_datasets/map/pretrain_mimic/pretrain_mimic_hf.json \
# --fold_json src/ecg_datasets/map/pretrain_mimic/pretrain_mimic_hf_folds.json \
# --repo_id willxxy/pretrain-mimic-250-2500


# uv run src/upload_hf.py \
# --map_json src/ecg_datasets/map/ecg_qa/ecg_qa_mimic_iv_hf.json \
# --fold_json src/ecg_datasets/map/ecg_qa/ecg_qa_mimic_iv_hf_folds.json \
# --repo_id willxxy/ecg-qa-mimic-iv-ecg-250-2500

# uv run src/upload_hf.py \
# --map_json src/ecg_datasets/map/ecg_qa/ecg_qa_ptb_xl_hf.json \
# --fold_json src/ecg_datasets/map/ecg_qa/ecg_qa_ptb_xl_hf_folds.json \
# --repo_id willxxy/ecg-qa-ptbxl-ecg-250-2500


# uv run src/upload_hf.py \
# --map_json src/ecg_datasets/map/ecg_instruct_45k/ecg_instruct_45k_hf.json \
# --fold_json src/ecg_datasets/map/ecg_instruct_45k/ecg_instruct_45k_hf_folds.json \
# --repo_id willxxy/ecg-instruct-45k-250-2500


# uv run src/upload_hf.py \
# --map_json src/ecg_datasets/map/ecg_bench_pulse/ecg_bench_pulse_hf.json \
# --fold_json src/ecg_datasets/map/ecg_bench_pulse/ecg_bench_pulse_hf_folds.json \
# --repo_id willxxy/ecg-bench-pulse-250-2500


# uv run src/upload_hf.py \
# --map_json src/ecg_datasets/map/ecg_instruct_pulse/ecg_instruct_pulse_hf.json \
# --fold_json src/ecg_datasets/map/ecg_instruct_pulse/ecg_instruct_pulse_hf_folds.json \
# --repo_id willxxy/ecg-instruct-pulse-250-2500

uv run src/upload_hf.py \
--map_json src/ecg_datasets/map/ecg_grounding/ecg_grounding_hf.json \
--fold_json src/ecg_datasets/map/ecg_grounding/ecg_grounding_hf_folds.json \
--repo_id willxxy/ecg-grounding-250-2500