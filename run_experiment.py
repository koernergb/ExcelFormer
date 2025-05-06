import subprocess
import time
from datetime import datetime
import os
import numpy as np

DATASET = "android_security"
SIZES = [10000, 100000, None]  # None means full dataset
BASE_CMD = "python -u run_default_config_excel.py --dataset {dataset} --catenc --mix_type hidden_mix --save{sample_size_arg}"
EVAL_CMD = "python evaluate.py --model_path '{model_path}' --dataset {dataset} --catenc{sample_size_arg}"
PLOT_CMD = "python plot_training_metrics.py --log_file '{log_file}'{sample_size_arg}"

def get_model_path(size):
    size_str = "full" if size is None else str(size)
    return f"result/ExcelFormer/default/mixup(hidden_mix)/{DATASET}/42/{size_str}/pytorch_model.pt"

def main():
    log_files = []
    model_paths = []

    for size in SIZES:
        size_str = "full" if size is None else str(size)
        sample_size_arg = "" if size is None else f" --sample_size {size}"
        # Clean up any old checkpoints or temp files for this size
        model_path = get_model_path(size)
        if os.path.exists(model_path):
            os.remove(model_path)
        # Optionally, clean up logs or other artifacts here

        print(f"\n=== Training with sample size: {size_str} ===")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_log_{size_str}_{timestamp}.txt"
        log_files.append(log_file)
        train_cmd = BASE_CMD.format(dataset=DATASET, sample_size_arg=sample_size_arg) + f" 2>&1 | tee {log_file}"
        print(f"Running: {train_cmd}")
        subprocess.run(train_cmd, shell=True, check=True)
        time.sleep(2)  # Give filesystem a moment

        model_path = get_model_path(size)
        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found at {model_path}")
        model_paths.append(model_path)

        print(f"Evaluating model: {model_path}")
        eval_cmd = EVAL_CMD.format(model_path=model_path, dataset=DATASET, sample_size_arg=sample_size_arg)
        subprocess.run(eval_cmd, shell=True, check=True)
        time.sleep(2)

    print("\n=== All experiments complete! ===")
    print("Training logs:", log_files)
    print("Model paths:", model_paths)
    print("Check for ROC curves and training metrics PNGs in your directory.")

if __name__ == "__main__":
    main()
