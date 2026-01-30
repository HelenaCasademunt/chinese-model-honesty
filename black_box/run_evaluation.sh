#!/bin/bash
# Run hypothesis_auditor on all converted response files.
# Assumes convert_responses.py has already been run.

CONVERTED_DIR="results_new_pipeline/converted"
HYPOTHESIS_OUT="results_new_pipeline/hypotheses"
GT_FILE="/root/chinese_auditing/data/tiananmen_square_1989.json"
JUDGE_MODEL="google/gemini-3-flash-preview"
MAX_CONCURRENT=50

mkdir -p "$HYPOTHESIS_OUT"

echo "========================================"
echo "Running hypothesis auditor"
echo "Converted dir: $CONVERTED_DIR"
echo "Ground truth: $GT_FILE"
echo "Judge model: $JUDGE_MODEL"
echo "========================================"

for f in "$CONVERTED_DIR"/*.json "$CONVERTED_DIR"/**/*.json; do
    [ -f "$f" ] || continue
    BASENAME=$(basename "$f" .json)
    RELDIR=$(dirname "$f" | sed "s|$CONVERTED_DIR||" | sed 's|^/||')

    if [ -n "$RELDIR" ]; then
        OUTDIR="$HYPOTHESIS_OUT/$RELDIR"
    else
        OUTDIR="$HYPOTHESIS_OUT"
    fi
    mkdir -p "$OUTDIR/$BASENAME"

    CONFIG_FILE=$(mktemp /tmp/hypothesis_XXXXXX.yaml)
    cat > "$CONFIG_FILE" <<EOF
input_file: "$f"
output_dir: "$OUTDIR/$BASENAME"
gt_file: "$GT_FILE"
model: "$JUDGE_MODEL"
fact_check_model: "$JUDGE_MODEL"
disable_reasoning: true
max_concurrent: $MAX_CONCURRENT
temperature: 0.3
max_tokens: 2000
metrics:
  model: "$JUDGE_MODEL"
  max_concurrent: $MAX_CONCURRENT
  disable_reasoning: true
EOF

    echo ""
    echo "  Hypothesis auditor: $f"
    python /root/chinese_auditing/src/hypothesis_auditor.py run "$CONFIG_FILE"
    rm -f "$CONFIG_FILE"
done

echo ""
echo "========================================"
echo "Evaluation complete. Results in: $HYPOTHESIS_OUT"
echo "========================================"
