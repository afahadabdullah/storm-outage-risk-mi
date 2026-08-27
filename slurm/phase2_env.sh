# Shared runtime for every Phase 2 batch job.
# Use an absolute interpreter path: login shells can discard an activated
# Conda environment before the first Python import.
PHASE2_ENV="/panfs/ccds02/nobackup/people/afahad/envs/storm-outage-risk"
PHASE2_PYTHON="${PHASE2_ENV}/bin/python"

if [[ ! -x "${PHASE2_PYTHON}" ]]; then
  printf 'Phase 2 Python is missing or not executable: %s\n' "${PHASE2_PYTHON}" >&2
  exit 2
fi

export PATH="${PHASE2_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1

