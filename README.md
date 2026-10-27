# README

This repository contains the replication code for our TMLR paper:

*Evaluating Ensemble Methods for Algorithmic Stability Under Data Perturbations*

https://openreview.net/forum?id=7CCr0dgNwm

If you have any questions, please reach out to benedel@uw.edu.

---

## 1. Setup & Environment

This code is implemented in Python 3.12.10. To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

All the experiments use fixed seeds for reproducibility and were run locally on a MacBook Air M3 with 16GB of memory.

## 2. Data

The verbal autopsy data from PHMRC can be found at:

https://ghdx.healthdata.org/record/ihme-data/population-health-metrics-research-consortium-gold-standard-verbal-autopsy-data-2005-2011

Navigate to the "Files" tab and download the `PHMRC VA adult data, CSV format`. Rename the file to `va_data_adult.csv` and place the file into each necessary directory.

The HELOC dataset from FICO can be found at the following link:

https://huggingface.co/datasets/mstz/heloc

Rename the CSV file to `heloc.csv` and place in the respective directory.

## 3. Running Experiments

#### E1: Verbal Autopsy - Leave-One-Out

To run the subbagging and inflated argmax experiment on the verbal autopsy dataset with leave-one-out perturbations execute the following commands within the `E1 - Verbal Autopsy - LOO` directory:

```bash
chmod +x parallel_jobs.sh
./parallel_jobs.sh
```

Then run the following to compile the results and generate the tables and figures:

```bash
python3 results.py
```

With default parameters, this takes approximately 20 minutes to run, though this can be significantly reduced with fewer bags and trials. The default validation site is "Dar". If you would like to change the validation site, this must be done in all three files (`parallel_jobs.sh`, `fit.py`, `results.py`).

This can be used to recreate Tables 1-2, 4-9, and 13, Figure 7b, and by extension Figures 1-2, 4, 6, 10, and 15.

#### E2: Verbal Autopsy - Multi-Point Perturbations

To run the subbagging and inflated argmax experiment on the verbal autopsy dataset with bootstrap perturbations execute the following commands within the `E2 - Verbal Autopsy - MPP` directory:

```bash
chmod +x parallel_jobs_mpp.sh
./parallel_jobs_mpp.sh
```

Then run the following to compile the results and generate the tables and figures:

```bash
python3 results_mpp.py
```

The default model is LDA and the default perturbation scheme is bootstrap, though these can be changed as needed in `parallel_jobs_mpp.sh`. The alternative perturbation scheme is leave-k-out and the k-parameter can be changed from the default k=1000 in `fit_mpp.py`. The default validation site is "Dar". If you would like to change the validation site, this must be done in `fit_mpp.py` and `results_mpp.py`. This takes about 30 minutes to run with default parameters.

This is used to recreate Figures 3, 8, and 9.

#### E3: Verbal Autopsy - InSilicoVA

InSilicoVA needs to be run with R. This requires a local installation of R and the `openVA` and `InSilicoVA` packages. Before running the Python scripts, ensure these are installed via your R console:

```R
install.packages(c("openVA", "InSilicoVA"))
```

To run the subbagging and inflated argmax experiment on the verbal autopsy dataset with InSilicoVA and leave-one-out perturbations execute the following commands within the `E3 - Verbal Autopsy - InSilicoVA` directory:

```bash
chmod +x parallel_jobs_openva.sh
./parallel_jobs_openva.sh
```

Then run the following to compile the results and generate the tables and figures:

```bash
python3 results_openva.py
```

The InSilicoVA pre-processing is entirely different so it requires separate files to run. It is also much slower and thus takes quite some time to run (over 2.5 hours with default parameters), so we recommend only running this in full if absolutely needed, otherwise reduce the number of bags and trials.

This is used to recreate Table 3 and Figure 7a.

#### E4: Rashomon Ensemble

To run the Rashomon experiment execute the following commands within the `E4 - Rashomon Ensemble` directory:

```bash
chmod +x parallel_jobs_rashomon.sh
./parallel_jobs_rashomon.sh
```

Then run the following to compile the results and generate the tables and figures:

```bash
python3 results_rashomon.py synthetic
```

The default dataset is synthetic and the default perturbation scheme is bootstrap, both of which can be changed in the `parallel_jobs_rashomon.sh` file, as well as the dataset in the last command. With default parameters, the synthetic experiment takes approximately 15 minutes to run and the HELOC experiment takes approximately 1 hour to run.

This is used to recreate Tables 10-11, 14-16 and Figures 5, 11-14.

##### Acknowledgments

We would like to thank Soloff et al. (2024), whose work developing the inflated argmax method provides the basis for most of this code. We would also like to acknowledge the work of Arslan et al. (2025), who provided the `pysortd` package to allow for efficient Rashomon enumeration that made these experiments feasible. 
