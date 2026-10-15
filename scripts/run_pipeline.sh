#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:-sentinel}"
cd "${PROJECT_ROOT}" || exit 1
# MODEL, MODEL_PATH, PROJECTOR_MODEL and PROJECTOR_MODEL_PATH, together with all
# scientific settings, are resolved once and exported by the configuration module.
# Atomic reservation happens before any artifact write. No shared prompt is rewritten.
RESOLVED_EXPORTS="$("${PYTHON_BIN}" -m morace.run_configuration prepare \
  --project-root "${PROJECT_ROOT}" --target "${TARGET}")" || exit 1
eval "${RESOLVED_EXPORTS}"
export MORACE_SEED="${GLOBAL_SEED}"
LOG_DIR="artifacts/${RUN_TAG}/logs"
SUMMARY_PATH="artifacts/${RUN_TAG}/summary.json"
OVERALL_LOG_PATH="${LOG_DIR}/overall.log"
LAUNCH_SCRIPT_PATH="${SCRIPT_DIR}/run_pipeline.sh"
touch "${OVERALL_LOG_PATH}" || exit 1
echo "[Pipeline] run_tag=${RUN_TAG} full_log=${OVERALL_LOG_PATH}"
"${PYTHON_BIN}" -m morace.runtime_snapshot \
  --run-config "artifacts/${RUN_TAG}/run_config.json" \
  --synthesize-prompt "${PROMPT_DIR}/synthesize_prompt.json" \
  --inference-prompt "${PROMPT_DIR}/inference_prompt.json" \
  --launch-script "${LAUNCH_SCRIPT_PATH}" || exit 1

OTHERS_DATASETS=("bbbp" "bace" "clintox" "esol" "freesolv" "hiv" "lipophilicity")
TOX21_SUBTASKS=("nr-ar-lbd" "nr-ahr" "nr-ar" "nr-aromatase" "nr-er" "nr-er-lbd" "nr-ppar-gamma" "sr-are" "sr-atad5" "sr-hse" "sr-mmp" "sr-p53")
SIDER_SUBTASKS=("cardiac disorders" "neoplasms benign, malignant and unspecified (incl cysts and polyps)" "hepatobiliary disorders" "metabolism and nutrition disorders" "product issues" "eye disorders" "investigations" "musculoskeletal and connective tissue disorders" "gastrointestinal disorders" "social circumstances" "immune system disorders" "reproductive system and breast disorders" "general disorders and administration site conditions" "endocrine disorders" "surgical and medical procedures" "vascular disorders" "blood and lymphatic system disorders" "skin and subcutaneous tissue disorders" "congenital, familial and genetic disorders" "infections and infestations" "respiratory, thoracic and mediastinal disorders" "psychiatric disorders" "renal and urinary disorders" "pregnancy, puerperium and perinatal conditions" "ear and labyrinth disorders" "nervous system disorders" "injury, poisoning and procedural complications")
QM9_SUBTASKS=("alpha" "c_v" "Delta_epsilon" "epsilon_HOMO" "epsilon_LUMO" "G" "H" "mu" "R^2" "U_0" "U" "ZPVE")
SENTINEL_OTHERS=("bbbp" "bace" "esol")

sanitize_task_label() {
  local raw="$1"
  local cleaned
  cleaned="$(printf '%s' "${raw}" | tr ' /:()' '______' | tr -cs '[:alnum:]_.-' '_')"
  cleaned="${cleaned#_}"
  cleaned="${cleaned%_}"
  if [[ -z "${cleaned}" ]]; then
    cleaned="task"
  fi
  printf '%s' "${cleaned}"
}

PIPELINE_LAST_REASON=""
PIPELINE_FAILURE_COUNT=0

run_step() {
  local step="$1"
  shift
  echo "[Pipeline] step=${step} started"
  if "$@"; then
    echo "[Pipeline] step=${step} finished"
    return 0
  else
    local exit_code=$?
    PIPELINE_LAST_REASON="step=${step} exit_code=${exit_code}"
    echo "[Pipeline] step=${step} failed exit_code=${exit_code}"
    return "${exit_code}"
  fi
}

append_summary() {
  local dataset="$1"
  local subtask="$2"
  local batch_num="$3"
  local exec_status="$4"
  local exit_code="$5"
  local log_path="$6"
  local reason="${7:-}"
  "${PYTHON_BIN}" -m morace.run_summary \
    --summary-path "${SUMMARY_PATH}" \
    --run-tag "${RUN_TAG}" \
    --model "${MODEL}" \
    --dataset "${dataset}" \
    --subtask "${subtask}" \
    --batch-num "${batch_num}" \
    --exec-status "${exec_status}" \
    --exit-code "${exit_code}" \
    --log-path "${log_path}" \
    --reason "${reason}"
}

run_pipeline() {
  local dataset="$1"
  local subtask="$2"
  local batch_num="$3"
  local task_label="$dataset"
  if [[ -n "${subtask}" ]]; then
    task_label="${dataset}:${subtask}"
  fi
  echo "=================================================="
  echo "[Pipeline] Start task=${task_label} model=${MODEL} projector=${PROJECTOR_MODEL} batch_num=${batch_num} run_tag=${RUN_TAG}"
  echo "[Pipeline] Settings: syn_drafts_per_prompt=${SYN_DRAFTS_PER_PROMPT} inf_drafts_per_prompt=${INF_DRAFTS_PER_PROMPT} max_samples_per_chunk=${MAX_SAMPLES_PER_CHUNK}"
  echo "[Pipeline] ResolvedConfig project_root=${PROJECT_ROOT}"
  echo "[Pipeline] ResolvedConfig model_path=${MODEL_PATH}"
  echo "[Pipeline] ResolvedConfig projector_model=${PROJECTOR_MODEL}"
  echo "[Pipeline] ResolvedConfig projector_path=${PROJECTOR_MODEL_PATH}"
  echo "[Pipeline] ResolvedConfig use_cross_model_projector=${USE_CROSS_MODEL_PROJECTOR}"
  echo "[Pipeline] ResolvedConfig launcher_default_projector_model=${LAUNCHER_DEFAULT_PROJECTOR_MODEL}"
  echo "[Pipeline] ResolvedConfig schema_mode=${SCHEMA_MODE}"
  echo "[Pipeline] ResolvedConfig heuristic_policy=${HEURISTIC_POLICY}"
  echo "[Pipeline] ResolvedConfig stage2_backend=${STAGE2_BACKEND}"
  echo "[Pipeline] ResolvedConfig stage2_protocol_profile_override=${STAGE2_PROTOCOL_PROFILE}"
  echo "[Pipeline] ResolvedConfig skip_prompt_regen=${SKIP_PROMPT_REGEN}"
  date
  PIPELINE_LAST_REASON=""

  echo "[Pipeline] dataset_provenance started"
  if "${PYTHON_BIN}" -m morace.dataset_provenance \
    --run-config "artifacts/${RUN_TAG}/run_config.json" \
    --dataset "${dataset}" \
    --subtask "${subtask}" \
    --project-root "${PROJECT_ROOT}" \
    --manifest "${DATASET_MANIFEST}"; then
    echo "[Pipeline] dataset_provenance finished"
  else
    local provenance_exit=$?
    PIPELINE_LAST_REASON="step=dataset_provenance exit_code=${provenance_exit}"
    echo "[Pipeline] dataset_provenance failed exit_code=${provenance_exit}"
    return "${provenance_exit}"
  fi

  run_step synthesize "${PYTHON_BIN}" -m morace.synthesis \
    --input_folder "${PROMPT_DIR}" \
    --dataset "${dataset}" \
    --subtask "${subtask}" \
    --model "${MODEL}" \
    --model_path "${MODEL_PATH}" \
    --seed "${GLOBAL_SEED}" \
    --projector_model "${PROJECTOR_MODEL}" \
    --projector_model_path "${PROJECTOR_MODEL_PATH}" \
    --extra_draft_models "${EXTRA_DRAFT_MODELS}" \
    --extra_draft_model_paths "${EXTRA_DRAFT_MODEL_PATHS}" \
    --drafts_per_prompt "${SYN_DRAFTS_PER_PROMPT}" \
    --max_new_tokens "${SYN_MAX_NEW_TOKENS}" \
    --projector_max_new_tokens "${SYN_PROJECTOR_MAX_NEW_TOKENS}" \
    --max_prompt_tokens "${SYN_MAX_PROMPT_TOKENS}" \
    --projector_max_prompt_tokens "${PROJECTOR_MAX_PROMPT_TOKENS}" \
    --stage1_temperature "${STAGE1_TEMPERATURE}" \
    --stage2_temperature "${STAGE2_TEMPERATURE}" \
    --stage1_top_p "${STAGE1_TOP_P}" \
    --stage2_top_p "${STAGE2_TOP_P}" \
    --stage1_num_beams "${STAGE1_NUM_BEAMS}" \
    --stage2_num_beams "${STAGE2_NUM_BEAMS}" \
    --run_tag "${RUN_TAG}" || { local step_exit=$?; return "${step_exit}"; }

  run_step inference "${PYTHON_BIN}" -m morace.inference \
    --prompt_folder "${PROMPT_DIR}" \
    --dataset "${dataset}" \
    --subtask "${subtask}" \
    --list_num "${batch_num}" \
    --max_samples_per_chunk "${MAX_SAMPLES_PER_CHUNK}" \
    --seed "${GLOBAL_SEED}" \
    --model "${MODEL}" \
    --model_path "${MODEL_PATH}" \
    --projector_model "${PROJECTOR_MODEL}" \
    --projector_model_path "${PROJECTOR_MODEL_PATH}" \
    --extra_draft_models "${EXTRA_DRAFT_MODELS}" \
    --extra_draft_model_paths "${EXTRA_DRAFT_MODEL_PATHS}" \
    --drafts_per_prompt "${INF_DRAFTS_PER_PROMPT}" \
    --max_new_tokens "${INF_MAX_NEW_TOKENS}" \
    --projector_max_new_tokens "${INF_PROJECTOR_MAX_NEW_TOKENS}" \
    --max_prompt_tokens "${INF_MAX_PROMPT_TOKENS}" \
    --projector_max_prompt_tokens "${PROJECTOR_MAX_PROMPT_TOKENS}" \
    --stage1_temperature "${STAGE1_TEMPERATURE}" \
    --stage2_temperature "${STAGE2_TEMPERATURE}" \
    --stage1_top_p "${STAGE1_TOP_P}" \
    --stage2_top_p "${STAGE2_TOP_P}" \
    --stage1_num_beams "${STAGE1_NUM_BEAMS}" \
    --stage2_num_beams "${STAGE2_NUM_BEAMS}" \
    --run_tag "${RUN_TAG}" || { local step_exit=$?; return "${step_exit}"; }

  run_step summarize "${PYTHON_BIN}" -m morace.rule_summary \
    --input_model_folder "${MODEL}" \
    --dataset "${dataset}" \
    --subtask "${subtask}" \
    --list_num "${batch_num}" \
    --target_rule_count "${TARGET_RULE_COUNT}" \
    --run_tag "${RUN_TAG}" || { local step_exit=$?; return "${step_exit}"; }

  run_step train_eval_synthesize "${PYTHON_BIN}" -m morace.evaluation_pipeline \
    --dataset "${dataset}" \
    --subtask "${subtask}" \
    --model "${MODEL}" \
    --knowledge_type "synthesize" \
    --list_num "${batch_num}" \
    --target_rule_count "${TARGET_RULE_COUNT}" \
    --generation_seed "${GLOBAL_SEED}" \
    --run_tag "${RUN_TAG}" || { local step_exit=$?; return "${step_exit}"; }

  run_step train_eval_inference "${PYTHON_BIN}" -m morace.evaluation_pipeline \
    --dataset "${dataset}" \
    --subtask "${subtask}" \
    --model "${MODEL}" \
    --knowledge_type "inference" \
    --list_num "${batch_num}" \
    --target_rule_count "${TARGET_RULE_COUNT}" \
    --generation_seed "${GLOBAL_SEED}" \
    --run_tag "${RUN_TAG}" || { local step_exit=$?; return "${step_exit}"; }

  run_step train_eval_all "${PYTHON_BIN}" -m morace.evaluation_pipeline \
    --dataset "${dataset}" \
    --subtask "${subtask}" \
    --model "${MODEL}" \
    --knowledge_type "all" \
    --list_num "${batch_num}" \
    --target_rule_count "${TARGET_RULE_COUNT}" \
    --generation_seed "${GLOBAL_SEED}" \
    --run_tag "${RUN_TAG}" || { local step_exit=$?; return "${step_exit}"; }

  echo "[Pipeline] Finished task=${task_label}"
  date
}

run_pipeline_logged() {
  local dataset="$1"
  local subtask="$2"
  local batch_num="$3"
  local task_label="${dataset}"
  if [[ -n "${subtask}" ]]; then
    task_label="${dataset}:${subtask}"
  fi
  local task_slug
  task_slug="$(sanitize_task_label "${task_label}")"
  local log_path="${LOG_DIR}/${task_slug}.log"
  local exit_code=0
  local task_status=""

  echo "=================================================="
  echo "[Pipeline] Launch task=${task_label} log=${log_path}"
  if run_pipeline "${dataset}" "${subtask}" "${batch_num}" > >(tee "${log_path}" -a "${OVERALL_LOG_PATH}") 2>&1; then
    task_status="$(append_summary "${dataset}" "${subtask}" "${batch_num}" "success" "0" "${log_path}" "")"
    echo "[Pipeline] Completed task=${task_label} status=${task_status} log=${log_path}"
    if [[ "${task_status}" != "success" ]]; then
      PIPELINE_FAILURE_COUNT=$((PIPELINE_FAILURE_COUNT + 1))
    fi
  else
    exit_code=$?
    task_status="$(append_summary "${dataset}" "${subtask}" "${batch_num}" "failed" "${exit_code}" "${log_path}" "${PIPELINE_LAST_REASON}")"
    echo "[Pipeline] Completed task=${task_label} status=${task_status} exit_code=${exit_code} log=${log_path}"
    echo "[Pipeline] Failure summary for ${task_label}: ${PIPELINE_LAST_REASON}"
    tail -n 20 "${log_path}" || true
    PIPELINE_FAILURE_COUNT=$((PIPELINE_FAILURE_COUNT + 1))
  fi
}

run_target() {
  local target="$1"
  case "${target}" in
    bbbp|bace|clintox|esol|freesolv|hiv|lipophilicity)
      run_pipeline_logged "${target}" "" "${LIST_NUM}"
      ;;
    sentinel)
      for dataset in "${SENTINEL_OTHERS[@]}"; do
        run_pipeline_logged "${dataset}" "" "${LIST_NUM}"
      done
      ;;
    others)
      for dataset in "${OTHERS_DATASETS[@]}"; do
        run_pipeline_logged "${dataset}" "" "${LIST_NUM}"
      done
      ;;
    tox21)
      for subtask in "${TOX21_SUBTASKS[@]}"; do
        run_pipeline_logged "tox21" "${subtask}" "${LIST_NUM}"
      done
      ;;
    sider)
      for subtask in "${SIDER_SUBTASKS[@]}"; do
        run_pipeline_logged "sider" "${subtask}" "${LIST_NUM}"
      done
      ;;
    qm9)
      for subtask in "${QM9_SUBTASKS[@]}"; do
        run_pipeline_logged "qm9" "${subtask}" "${QM9_LIST_NUM}"
      done
      ;;
    all)
      run_target others
      ;;
    tox21:*)
      run_pipeline_logged "tox21" "${target#tox21:}" "${LIST_NUM}"
      ;;
    sider:*)
      run_pipeline_logged "sider" "${target#sider:}" "${LIST_NUM}"
      ;;
    qm9:*)
      run_pipeline_logged "qm9" "${target#qm9:}" "${QM9_LIST_NUM}"
      ;;
    *)
      echo "Unsupported target: ${target}"
      echo "Usage: bash scripts/run_pipeline.sh [bbbp|bace|clintox|esol|freesolv|hiv|lipophilicity|sentinel|others|tox21|sider|qm9|all|tox21:<subtask>|sider:<subtask>|qm9:<subtask>]"
      exit 1
      ;;
  esac
}

run_target "${TARGET}"

"${PYTHON_BIN}" - "${SUMMARY_PATH}" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

print("==================================================")
print(f"[Pipeline] Final summary saved to {summary_path}")
for key, value in sorted(payload.get("counts", {}).items()):
    print(f"[Pipeline] {key}={value}")
for item in payload.get("tasks", []):
    print(
        f"[Pipeline] task={item['task_label']} status={item['status']} "
        f"exit_code={item['exit_code']} log={item['log_path']}"
    )
PY

"${PYTHON_BIN}" -m morace.run_configuration finish \
  --project-root "${PROJECT_ROOT}" --run-tag "${RUN_TAG}"
exit $?
