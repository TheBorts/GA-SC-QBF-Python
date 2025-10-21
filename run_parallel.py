#!/usr/bin/env python3
# run_experiments_python.py
"""
Runs GA_SCQBF.py experiments with multiple parameter combinations and instances.
Translated from the Java-runner version to work with the Python implementation.
"""

import subprocess
import os
import glob
import concurrent.futures
from pathlib import Path

# Configuration
GA_SCRIPT = "src/ga_scqbf.py"   # Path to your translated Python GA solver
INSTANCES_DIR = "instances"
RESULTS_DIR = "results"
LOG_DIR = "logs"

# Parameters to test
PARAM_COMBINATIONS = [
    {
        "enable_latin_hyper_cube": False,
        "enable_mutate_or_crossover": True,
        "enable_uniform_crossover": True,
        "population_size": 100,
        "mutation_rate": 0.05,
        "time_limit": 30
    }
]


def run_experiment(instance_file, params):
    """Run a single experiment with given parameters using GA_SCQBF.py"""
    instance_name = Path(instance_file).stem
    param_str = (
        f"l{params['enable_latin_hyper_cube']}_"
        f"m{params['enable_mutate_or_crossover']}_"
        f"u{params['enable_uniform_crossover']}_"
        f"{params['population_size']}"
    )

    log_file = os.path.join(LOG_DIR, f"{instance_name}_{param_str}.log")
    result_file = os.path.join(RESULTS_DIR, f"{instance_name}_{param_str}.txt")

    cmd = [
        "python3", GA_SCRIPT,
        "--instance", instance_file,
        "--enable-latin-hyper-cube", str(params["enable_latin_hyper_cube"]).lower(),
        "--enable-mutate-or-crossover", str(params["enable_mutate_or_crossover"]).lower(),
        "--enable-uniform-crossover", str(params["enable_uniform_crossover"]).lower(),
        "--tl", str(params["time_limit"]),
        "--population-size", str(params["population_size"]),
        "--mutation-rate", str(params["mutation_rate"])
    ]

    print(f"Running: {instance_name} with {param_str}")

    try:
        with open(log_file, "w") as log_f:
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=params["time_limit"] + 60)

        # Extract key results
        with open(log_file, "r") as log_f:
            log_content = log_f.read()

        with open(result_file, "w") as res_f:
            res_f.write(f"Instance: {instance_name}\n")
            res_f.write(f"Parameters: {params}\n\n")
            for line in log_content.splitlines():
                if any(keyword in line for keyword in ["maxVal =", "TotalTime", "BestSol =", "Real value ="]):
                    res_f.write(line + "\n")

        return f"✅ Completed: {instance_name}_{param_str}"

    except subprocess.TimeoutExpired:
        return f"⏱️ TIMEOUT: {instance_name}_{param_str}"
    except Exception as e:
        return f"❌ ERROR: {instance_name}_{param_str} - {str(e)}"


def main():
    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Find instance files
    instance_files = glob.glob(os.path.join(INSTANCES_DIR, "*.txt"))
    if not instance_files:
        print(f"No instance files found in {INSTANCES_DIR}")
        return

    print(f"Found {len(instance_files)} instance files")

    # Prepare experiment list
    experiments = [(instance, params) for instance in instance_files for params in PARAM_COMBINATIONS]
    print(f"Running {len(experiments)} experiments...")

    # Run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_experiment, inst, param) for inst, param in experiments]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
