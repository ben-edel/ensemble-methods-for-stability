#!/bin/bash
# Run with:
# chmod +x parallel_jobs.sh
# ./parallel_jobs.sh

full_start=$(date +%s)

bags=1200
trials=500
max_jobs=6

# Set the perturbation scheme and dataset
model="mlp" # Options: "mlp", "gradient_boosting", "logistic_regression", "lda", "naive_bayes", "knn", "decision_tree", "reg_decision_tree"

echo "Training full model..."
python3 fit.py full 1 "$model"

echo "Training $bags subbagging models in parallel (max $max_jobs jobs)..."
for ((i = 1; i <= bags; i++)); do
    echo "Starting subbagging model $i"
    python3 fit.py subbagging $i "$model" &
    
    # Limit number of parallel jobs
    if (( i % max_jobs == 0 )); then
        wait  # Wait for current batch to finish before starting next batch
    fi
done
wait  # Wait for any remaining jobs to finish

echo "Training $trials LOO models in parallel (max $max_jobs jobs)..."
# Generate $trials random indices from training set
indices=($(python3 -c "
import pandas as pd
import random
random.seed(0)
df = pd.read_csv('va_data_adult.csv', low_memory=False)
sample_size = len(df[df.site!='Dar'])
print(' '.join(map(str, random.sample(range(sample_size), $trials))))
"))
for ((i = 1; i <= trials; i++)); do
    idx=${indices[$((i - 1))]}
    echo "Starting LOO model $i with index $idx"
    python3 fit.py loo $i "$model" $idx &
    
    # Limit number of parallel jobs
    if (( i % max_jobs == 0 )); then
        wait  # Wait for current batch to finish before starting next batch
    fi
done
wait  # Wait for any remaining jobs to finish

full_end=$(date +%s)
full_duration=$((full_end - full_start))
full_minutes=$((full_duration / 60))

echo "All models trained in ${full_minutes} minutes"