#!/bin/bash
source /root/.venv/bin/activate

# Step 1: Run confession evaluation (ask model if it lied)
python asking_the_model/ask_if_true_confession_balanced.py \
    --input results/data/balanced_qa_dataset.json \
    --output results/data_asking_model/asking_confession_balanced.json

# Step 2: Classify the responses (CONFESSION/NO_CONFESSION/OTHER)
python asking_the_model/evaluate_asking_results_confession.py \
    --input results/data_asking_model/asking_confession_balanced.json \
    --output results/data_asking_model/evaluated_asking_confession_balanced.json
