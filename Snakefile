# Snakemake workflow for Crop Disease Detection Pipeline
# Team jn — IS477 Final Project
#
# Usage:
#   snakemake --cores 1              # run full pipeline (profile -> clean -> test)
#   snakemake --cores 1 train        # also run training (slow, requires GPU)
#   snakemake -n                     # dry run to preview steps
#
# Prerequisites:
#   pip install -r requirements.txt
#   Dataset must be downloaded and extracted to data/ first (see data/README.md)

DATA_DIR = "data/Multi-Crop Disease Dataset/Multicrop Disease Dataset/Multicrop Disease Dataset"
PROFILE_OUT = "data/profile_results.json"
WEIGHTS_OUT = "data/class_weights.json"
MODEL_PT = "results/best.pt"


# Default target: profile + clean + evaluate using the pre-trained model
rule all:
    input:
        PROFILE_OUT,
        WEIGHTS_OUT,
        "results/eval_complete.flag"


# --------------------------------------------------------------------------- #
# Step 1: Profile the dataset
# --------------------------------------------------------------------------- #
rule profile:
    input:
        DATA_DIR
    output:
        PROFILE_OUT
    message:
        "Profiling dataset and running QA checks..."
    shell:
        "python scripts/profile_dataset.py"


# --------------------------------------------------------------------------- #
# Step 2: Compute class weights from the profile results
# --------------------------------------------------------------------------- #
rule clean:
    input:
        PROFILE_OUT
    output:
        WEIGHTS_OUT
    message:
        "Computing class weights to characterize imbalance..."
    shell:
        "python scripts/cleaning_data.py"


# --------------------------------------------------------------------------- #
# Step 3 (optional): Train the model from scratch
#   - Requires a CUDA GPU and ~2-4 hours of compute time
#   - The pre-trained model at results/best.pt is already included in the repo
#   - Run only if you want to reproduce training from scratch
# --------------------------------------------------------------------------- #
rule train:
    input:
        weights=WEIGHTS_OUT,
        yaml=DATA_DIR + "/data.yaml",
        base_model="scripts/yolo26n.pt"
    output:
        MODEL_PT
    message:
        "Training YOLOv8 model for 100 epochs (requires GPU, ~2-4 hours)..."
    shell:
        """
        python -c "
from ultralytics import YOLO
import shutil, os

model = YOLO('scripts/yolo26n.pt')
results = model.train(
    data='{DATA_DIR}/data.yaml',
    epochs=100,
    imgsz=640,
    device='cuda',
    batch=8,
    project='runs/detect',
    name='train'
)
best = 'runs/detect/train/weights/best.pt'
os.makedirs('results', exist_ok=True)
shutil.copy(best, 'results/best.pt')
print('Model saved to results/best.pt')
        "
        """.format(DATA_DIR=DATA_DIR)


# --------------------------------------------------------------------------- #
# Step 4: Evaluate the model on the test split
# --------------------------------------------------------------------------- #
rule evaluate:
    input:
        MODEL_PT
    output:
        touch("results/eval_complete.flag")
    message:
        "Evaluating model on test split..."
    shell:
        "python scripts/test_model.py"
