#!/bin/bash
# Full training and evaluation pipeline for honesty finetuning (assistant tokens only)
# Trains on followup and mixed datasets, then evaluates each
# Goals dataset is already trained, just needs directory rename and upload

# Configuration
BASE_MODEL="Qwen/Qwen3-32B"
BALANCED_QA_DATASET="results/data/balanced_qa_dataset.json"
EVAL_QUESTIONS="black_box/eval/prompts/finegrained_eval_questions_filtered.json"

# Evaluation settings
EVALUATOR_MODEL="google/gemini-3-flash-preview"
MAX_CONCURRENT=50

# HuggingFace settings
HF_REPO_PREFIX="hcasademunt"

# Activate virtual environment
source /root/.venv/bin/activate

echo "========================================="
echo "Starting assistant-only training and evaluation pipeline"
echo "========================================="

# Handle the already-trained goals model
echo ""
echo "========================================="
echo "Processing existing goals model"
echo "========================================="

OLD_GOALS_DIR="/workspace/qwen3-32b-lora-finetuned"
NEW_GOALS_DIR="/workspace/qwen3-32b-honesty-finetuned-goals"

if [ -d "$OLD_GOALS_DIR" ]; then
    echo "Renaming goals model directory..."
    mv "$OLD_GOALS_DIR" "$NEW_GOALS_DIR"
    echo "✓ Renamed to $NEW_GOALS_DIR"
elif [ -d "$NEW_GOALS_DIR" ]; then
    echo "✓ Goals model directory already in correct location: $NEW_GOALS_DIR"
else
    echo "Warning: Goals model directory not found at $OLD_GOALS_DIR or $NEW_GOALS_DIR"
fi

# Array of datasets to process (goals is already trained, will skip training step)
DATASETS=("goals" "followup" "mixed")

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========================================="
    echo "Processing dataset: $DATASET"
    echo "========================================="

    # Set output directories and file names
    MODEL_DIR="/workspace/qwen3-32b-honesty-finetuned-${DATASET}"
    RESULTS_PREFIX="honesty_training/results"

    # Set num-samples based on dataset
    # For mixed: 5000 from each (so 10000 total)
    # For followup: 10000 from single dataset
    if [ "$DATASET" = "mixed" ]; then
        NUM_SAMPLES=5000
    else
        NUM_SAMPLES=10000
    fi

    # Step 1: Finetune the model (assistant tokens only)
    if [ "$DATASET" = "goals" ]; then
        echo ""
        echo "[1/6] Skipping training for goals dataset (already trained)"
    else
        echo ""
        echo "[1/6] Finetuning model on $DATASET dataset (num_samples=$NUM_SAMPLES, assistant tokens only)..."
        python honesty_training/finetune_qwen3_32b_assistant_only.py \
            --dataset-mode "$DATASET" \
            --num-samples "$NUM_SAMPLES" \
            --output-dir "$MODEL_DIR" \
            --epochs 1 \
            --batch-size 4 \
            --grad-accum 4 \
            --lr 1e-05 \
            --max-seq-length 1024 \
            --lora-r 32 \
            --lora-alpha 64 \
            --save-steps 1000

        if [ $? -ne 0 ]; then
            echo "Error: Finetuning failed for $DATASET"
            continue
        fi
    fi

    # Step 2: Run eval_local to get model responses
    echo ""
    echo "[2/6] Running eval_local for $DATASET model..."
    python honesty_training/eval_local.py \
        --model "$BASE_MODEL" \
        --lora-adapter "$MODEL_DIR" \
        --questions "$EVAL_QUESTIONS" \
        --output "${RESULTS_PREFIX}/responses_${DATASET}.json" \
        --temperature 0.7 \
        --num-samples 10 \
        --max-tokens 3072 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --batch-size 8 \
        --disable-compile

    if [ $? -ne 0 ]; then
        echo "Error: eval_local failed for $DATASET"
        continue
    fi

    # Step 3: Evaluate the responses using OpenRouter
    echo ""
    echo "[3/6] Evaluating responses with OpenRouter for $DATASET model..."
    python black_box/eval/evaluate_openrouter.py \
        --attack-type baseline \
        --input "${RESULTS_PREFIX}/responses_${DATASET}.json" \
        --output "${RESULTS_PREFIX}/evaluated_responses_${DATASET}.json" \
        --evaluator-model "$EVALUATOR_MODEL" \
        --max-concurrent "$MAX_CONCURRENT"

    if [ $? -ne 0 ]; then
        echo "Error: evaluate_openrouter failed for $DATASET"
        continue
    fi

    # Step 4: Run ask_if_true_balanced_local
    echo ""
    echo "[4/6] Running ask_if_true_balanced_local for $DATASET model..."
    python honesty_training/ask_if_true_balanced_local.py \
        --model "$BASE_MODEL" \
        --lora-adapter "$MODEL_DIR" \
        --input "$BALANCED_QA_DATASET" \
        --output "${RESULTS_PREFIX}/asking_truthfulness_balanced_${DATASET}.json" \
        --temperature 0.7 \
        --max-tokens 1024 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --batch-size 100 \
        --disable-compile

    if [ $? -ne 0 ]; then
        echo "Error: ask_if_true_balanced_local failed for $DATASET"
        continue
    fi

    # Step 5: Run ask_if_true_confession_balanced_local
    echo ""
    echo "[5/6] Running ask_if_true_confession_balanced_local for $DATASET model..."
    python honesty_training/ask_if_true_confession_balanced_local.py \
        --model "$BASE_MODEL" \
        --lora-adapter "$MODEL_DIR" \
        --input "$BALANCED_QA_DATASET" \
        --output "${RESULTS_PREFIX}/asking_confession_balanced_${DATASET}.json" \
        --temperature 0.7 \
        --max-tokens 3072 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --batch-size 100 \
        --disable-compile

    if [ $? -ne 0 ]; then
        echo "Error: ask_if_true_confession_balanced_local failed for $DATASET"
        continue
    fi

    # Step 6: Evaluate confession results
    echo ""
    echo "[6/6] Evaluating confession results for $DATASET model..."
    python asking_the_model/evaluate_asking_results_confession.py \
        --input "${RESULTS_PREFIX}/asking_confession_balanced_${DATASET}.json" \
        --output "${RESULTS_PREFIX}/evaluated_confession_balanced_${DATASET}.json" \
        --evaluator-model gpt-5.2 \
        --max-concurrent 10

    if [ $? -ne 0 ]; then
        echo "Error: evaluate_asking_results_confession failed for $DATASET"
        continue
    fi

    echo ""
    echo "✓ Completed all steps for $DATASET dataset"
done

# Upload models to HuggingFace
echo ""
echo "========================================="
echo "Uploading models to HuggingFace"
echo "========================================="

# Upload all three models (goals, followup, mixed)
ALL_DATASETS=("goals" "followup" "mixed")

for DATASET in "${ALL_DATASETS[@]}"; do
    MODEL_DIR="/workspace/qwen3-32b-honesty-finetuned-${DATASET}"

    if [ -d "$MODEL_DIR" ]; then
        echo ""
        echo "Uploading $DATASET model..."
        python upload_to_hf.py \
            "$MODEL_DIR" \
            "${HF_REPO_PREFIX}/qwen3-32b-honesty-finetuned-${DATASET}"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully uploaded $DATASET model"
        else
            echo "✗ Failed to upload $DATASET model"
        fi
    else
        echo "✗ Model directory not found: $MODEL_DIR"
    fi
done

echo ""
echo "========================================="
echo "Pipeline complete!"
echo "========================================="
echo ""
echo "Results saved in honesty_training/results/ with suffixes:"
echo "  - responses_followup.json"
echo "  - responses_mixed.json"
echo "  - evaluated_responses_followup.json"
echo "  - evaluated_responses_mixed.json"
echo "  - asking_truthfulness_balanced_followup.json"
echo "  - asking_truthfulness_balanced_mixed.json"
echo "  - asking_confession_balanced_followup.json"
echo "  - asking_confession_balanced_mixed.json"
echo "  - evaluated_confession_balanced_followup.json"
echo "  - evaluated_confession_balanced_mixed.json"
echo ""
echo "Models uploaded to HuggingFace:"
echo "  - ${HF_REPO_PREFIX}/qwen3-32b-honesty-finetuned-goals"
echo "  - ${HF_REPO_PREFIX}/qwen3-32b-honesty-finetuned-followup"
echo "  - ${HF_REPO_PREFIX}/qwen3-32b-honesty-finetuned-mixed"
echo ""
