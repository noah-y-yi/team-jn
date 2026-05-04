from ultralytics import YOLO
from pathlib import Path

def run_predictions(model):
    return model.predict(
        source=str(TEST_DIR),
        save=True,              # saves to 'runs/detect/predict
        conf=0.25               # confidence threshold
    )

def evaluate_predictions(model):
    return model.val(
        data="../data/Multi-Crop Disease Dataset/Multicrop Disease Dataset/Multicrop Disease Dataset/data.yaml",
        split="test"
    )

if __name__ == '__main__':
    # load model
    model = YOLO("../results/best.pt")

    # define the test image dataset
    TEST_DIR = Path("../data/Multi-Crop Disease Dataset/Multicrop Disease Dataset/Multicrop Disease Dataset/test/images").resolve()

    """UNCOMMENT IF YOU WANT TO VIEW PHYSICAL BOUNDING BOXES"""
    # run predictions for manual review
    #results = run_predictions(model)

    # automated evaluate predictions
    metrics = evaluate_predictions(model)
    print("\n--- Final Test Results ---")
    print(f"Mean Average Precision (mAP50-95): {metrics.box.map:.4f}")
    print(f"Mean Average Precision (mAP50):    {metrics.box.map50:.4f}")
