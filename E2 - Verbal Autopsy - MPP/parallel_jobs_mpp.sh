#!/bin/bash
# Run with:
# chmod +x parallel_jobs_mpp.sh
# ./parallel_jobs_mpp.sh

trials=500
max_jobs=6

# Set the model and perturbation scheme
model="lda" # Options: "lda", "naive_bayes", "decision_tree", "reg_decision_tree"
perturbation_scheme="bootstrap" # Options: "bootstrap", "leave_k_out"

echo "Training full model..."
start=$(date +%s)
python3 fit_mpp.py full 1 "$model" "$perturbation_scheme"
end=$(date +%s)
duration=$((end - start))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "Full model trained in ${minutes}m ${seconds}s"

echo "Training $trials perturbed models in parallel (max $max_jobs jobs)..."
echo "Start time: $(date "+%H:%M:%S")"
for ((i = 1; i <= trials; i++)); do
    echo "Starting perturbed model $i"
    python3 fit_mpp.py perturbed $i "$model" "$perturbation_scheme" &
    
    # Limit number of parallel jobs
    if (( i % max_jobs == 0 )); then
        wait  # Wait for current batch to finish before starting next batch
    fi
done
wait  # Wait for any remaining jobs to finish

end=$(date +%s)
duration=$((end - start))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "All models trained in ${minutes}m ${seconds}s"