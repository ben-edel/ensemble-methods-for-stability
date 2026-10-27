#!/bin/bash
# Run with:
# chmod +x parallel_jobs_rashomon.sh
# ./parallel_jobs_rashomon.sh

trials=200
max_jobs=6

# Set the perturbation scheme and dataset
perturbation_scheme="bootstrap" # Options: "bootstrap", "leave_k_out", "loo"
dataset="synthetic" # Options: "synthetic", "heloc"

echo "Training full model..."
start=$(date +%s)
python3 fit_rashomon.py full 1 "$perturbation_scheme" "$dataset"
end=$(date +%s)
duration=$((end - start))
minutes=$((duration / 60))
seconds=$((duration % 60))
echo "Full model trained in ${minutes}m ${seconds}s"

echo "Training $trials perturbed models in parallel (max $max_jobs jobs)..."
echo "Start time: $(date "+%H:%M:%S")"
for ((i = 1; i <= trials; i++)); do
    echo "Starting perturbed model $i"
    python3 fit_rashomon.py perturbed $i "$perturbation_scheme" "$dataset" &
    
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