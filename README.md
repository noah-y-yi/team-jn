# Crop Disease Detection Using Multi-Crop Computer Vision

## Contributors

* Noah Yi (University of Illinois Urbana-Champaign)  
* Jason Wu (University of Illinois Urbana-Champaign)

---

## Summary

Agricultural success and productivity is under the constant threat of plant diseases, which often cause an estimated 10 – 40% loss in global crop yields annually. Early and accurate identification of these diseases is critical for timely intervention, yet current detection methods rely heavily on the manual inspection trained by agronomists, a process that is slow, expensive, and largely inaccessible to rural farming communities in developing regions. Automated computer vision systems offer a scalable alternative, capable of analyzing leaf imagery in seconds and flagging diseased samples before infections can spread.

This project investigates whether a single unified object detection model can be trained to identify and localize 30 distinct disease and health conditions across five agriculturally important crops: Banana, Chilli, Radish, Groundnut, and Cauliflower. Our two central research questions are: (1) Can a unified YOLOv8 model achieve reliable multi-class detection performance across multiple crop types simultaneously? (2) How does class imbalance in training data affect per-class detection accuracy, and how should it be addressed during preprocessing?

We leverage the Multi-Crop Disease Dataset published on Mendeley Data (Prem Kumar E, VIT Chennai, 2024), a comprehensive collection of 21,875 annotated images captured from real farming fields across Tamil Nadu, India. The dataset provides 42,667 bounding boxes annotated across 30 classes in YOLO format, making it directly compatible with modern object detection frameworks without requiring format conversion.

Our pipeline consisted of four phases. First, we profiled the dataset using a custom script that verified image-label pairing integrity, counted per-class annotation distributions, and flagged any structural quality issues. Second, we performed data cleaning, primarily computing inverse-frequency class weights to characterize the severity of class imbalance discovered in profiling. Third, we fine-tuned a pretrained YOLO nano model using the Ultralytics library, training for 100 epochs on CUDA hardware with an input resolution of 640×640 pixels and a batch size of 8\. Fourth, we evaluated the trained model on the held-out test split of 2,191 images.

The model achieved an overall mean average precision (mAP50) of **0.714** and mAP50-95 of **0.515** across all 30 classes. Performance was strong for visually distinctive classes: *groundnut\_rust* achieved a perfect mAP50 of 0.995, *cauliflower\_healthy* and *banana\_healthy* both reached 0.995, and *banana\_bract\_mosaic\_virus* reached 0.974. Conversely, classes with high visual similarity to one another or very few training samples performed significantly below average: *chilli\_anthracnose* scored an mAP50 of only 0.030 and *chilli\_leafcurl* scored 0.113.

These findings confirm that multi-crop disease detection is feasible with a single model, and that pretrained YOLO weights transfer well to agricultural leaf imagery. However, the results also reveal that inter-class visual similarity within the Chilli crop, where multiple diseases produce superficially similar lesion patterns, and extreme data scarcity in several Cauliflower and Chilli classes remain key obstacles to production-level accuracy. The gap between mAP50 and mAP50-95 (0.714 vs. 0.515) indicates that the model correctly identifies disease presence but struggles with tight bounding box localization, a limitation that targeted augmentation and higher-resolution training could address in future iterations.  

---

## Data Profile

### Primary Dataset: Multi-Crop Disease Dataset

**Source:** Mendeley Data — [https://data.mendeley.com/datasets/6243z8r6t6/1](https://data.mendeley.com/datasets/6243z8r6t6/1)  
**Publisher:** Prem Kumar E, VIT Chennai  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Version:** 1  
**Location in repository:** Raw image data is excluded from version control due to GitHub's 100 MB file size limit. The dataset (\~978 MB zip) must be downloaded manually; full acquisition instructions are in [`data/`README.md](https://github.com/noah-y-yi/team-jn/blob/main/data/README.md). Profiling results derived from the dataset are stored at [data/profile\_results.json](https://github.com/noah-y-yi/team-jn/blob/main/data/profile_results.json).

#### Structure and Content:

The dataset contains annotated images of healthy and diseased leaves from five crops collected from real agricultural fields in Chengalpattu, Kanchipuram, and Krishnagiri districts of Tamil Nadu, India between November 2023 and January 2024\. Images were captured using high-resolution digital cameras and 200 MP mobile phones under natural field lighting conditions.

After extraction, the dataset follows standard YOLO directory conventions:

```
data/Multi-Crop Disease Dataset/Multicrop Disease Dataset/Multicrop Disease Dataset/
├── data.yaml           # class names and split paths
├── train/
│   ├── images/         # 15,310 images
│   └── labels/         # 15,310 YOLO .txt annotation files
├── valid/
│   ├── images/         # 4,374 images
│   └── labels/         # 4,374 YOLO .txt annotation files
└── test/
    ├── images/         # 2,191 images
    └── labels/         # 2,191 YOLO .txt annotation files
```

Each YOLO annotation file contains one row per object in the format: class\_id center\_x center\_y width height (all normalized to \[0, 1\] relative to image dimensions). Images are .jpg format at varying resolutions, downsampled to 640×640 during training.

**Split summary:**

| Split | Images | Annotations |
| :---- | :---- | :---- |
| Train | 15,310 | 30,481 |
| Valid | 4,374 | 7,881 |
| Test | 2,191 | 4,305 |
| **Total** | **21,875** | **42,667** |

The train/valid/test ratio is approximately 70%/20%/10%, consistent with standard practice for detection model development.

#### **Classes (30 total)**

The dataset covers 30 distinct disease and health-state classes across five crops:

| Crop | Classes |
| :---- | :---- |
| Banana | bract\_mosaic\_virus, cordana, healthy, insectpest, moko, panama, pestalotiopsis, sigatoka, yb\_sigatoka |
| Cauliflower | Blackrot, bacterial\_spot\_rot, downy\_mildew, healthy |
| Chilli | anthracnose, healthy, leafcurl, leafspot, whitefly, yellowish |
| Groundnut | early\_leaf\_spot, early\_rust, healthy, late\_leaf\_spot, nutrition\_deficiency, rust |
| Radish | black\_leaf\_spot, downey\_mildew, flea\_beetle, healthy, mosaic |

#### **Class Distribution**

A significant imbalance exists across classes. The full distribution is recorded in [data/profile\_results.json](https://github.com/noah-y-yi/team-jn/blob/main/data/profile_results.json). 

Highlights:

| Class | Total Annotations |
| :---- | :---- |
| banana\_sigatoka | 6,692 |
| groundnut\_early\_leaf\_spot | 6,323 |
| banana\_yb\_sigatoka | 4,004 |
| cauliflower\_Blackrot | 97 |
| chilli\_yellowish | 162 |
| cauliflower\_bacterial\_spot\_rot | 168 |

The most annotated class has approximately 69× more examples than the least annotated class.

#### **Ethical and Legal Constraints**

The dataset is published under CC BY 4.0, which permits use, redistribution, and adaptation for any purpose, including commercial, provided attribution is given to the original creators. The data was collected from farmland in Tamil Nadu, India; no personally identifiable information is present in the images. There are no known export restrictions or institutional data access agreements that limit use of this dataset in academic research.

The dataset cannot be re-distributed in its original form through the GitHub repository due to file size constraints. The [data/README.md](https://github.com/noah-y-yi/team-jn/blob/main/data/README.md) file provides a direct download link and SHA-256 integrity verification steps to ensure reproducibility.

#### **Relation to Research Questions**

This dataset directly addresses both research questions. Its multi-crop structure allows training and evaluating a single unified model across 5 crop types simultaneously. Its native YOLO annotation format provides bounding box labels that enable object detection (not merely classification), allowing the model to localize disease regions within leaves rather than labeling whole images. The documented class imbalance provides a concrete case study for evaluating the impact of imbalance on per-class detection performance.

### Secondary / External Datasets

Five external datasets were identified to support generalization testing across different geographic sources, camera conditions, and acquisition methods. These datasets were catalogued in [data/README.md](https://github.com/noah-y-yi/team-jn/blob/main/data/README.md) but were not acquired or integrated in this project due to time and scope constraints. External testing remains planned future work.

| Crop | Dataset | Source |
| :---- | :---- | :---- |
| Banana | Black Sigatoka & Fusarium Wilt | Harvard Dataverse |
| Chilli | Bacterial Spot | Mendeley |
| Radish | Black Leaf Spot | Mendeley |
| Groundnut | Rust & Leaf Spot | Mendeley |
| Cauliflower | Downy Mildew | Mendeley |

---

## Data Quality

Data quality assessment was performed using a custom profiling script ([`scripts/profile_dataset.py`](https://github.com/noah-y-yi/team-jn/blob/main/scripts/profile_dataset.py)) run against the extracted dataset. The script checks all three splits for: image-label file pairing completeness, zero-byte or unreadable image files, class ID coverage, and annotation count integrity. Full results are saved to [`data/profile_results.json`](https://github.com/noah-y-yi/team-jn/blob/main/data/profile_results.json).

### Structural Integrity

No structural QA issues were detected:

* **Image-label pairing:** All 21,875 images have a corresponding YOLO label file. Zero images are missing labels; zero label files are missing images.  
* **File integrity:** No zero-byte or empty image files were found across any split.  
* **Class coverage:** All 30 class IDs (0–29) appear in every split, train, valid, and test.  
* **Annotation format:** All label files follow the expected YOLO format without parse errors.

Despite the clean structural profile, two quality issues were identified at the content level:

### Issue 1: Severe Class Imbalance

The most significant quality concern is a 69:1 annotation ratio between the dominant and rarest classes. This imbalance is not uniform across crops — it is concentrated in the Cauliflower crop (all four of its classes fall below 210 total annotations) and in two Chilli classes (`chilli_yellowish`: 162, `chilli_whitefly`: 239).

Classes with fewer than 300 total annotations:

| Class ID | Class Name | Total Annotations |
| :---- | :---- | :---- |
| 9 | cauliflower\_Blackrot | 97 |
| 18 | chilli\_yellowish | 162 |
| 10 | cauliflower\_bacterial\_spot\_rot | 168 |
| 11 | cauliflower\_downy\_mildew | 186 |
| 12 | cauliflower\_healthy | 207 |
| 17 | chilli\_whitefly | 239 |
| 15 | chilli\_leafcurl | 297 |

This imbalance poses a direct risk of model bias toward frequently represented classes during training, suppressing the gradient signal from underrepresented classes. Without intervention, a model may learn to ignore minority classes because misclassifying them has negligible impact on the overall loss value.

### Issue 2: Inconsistent Class Name Formatting

Two class names in the source [`data.yaml`](https://github.com/noah-y-yi/team-jn/blob/main/data/data.yaml) contain formatting inconsistencies:

* `cauliflower_bacterial _spot _rot` — contains extra whitespace characters embedded within the name, which would cause string-matching errors if names are used as lookup keys in code.  
* `cauliflower_Blackrot` — uses a capital letter (`B`) inconsistent with all other class names, which follow lowercase snake\_case convention.

These issues do not affect model training (which uses integer class IDs, not string names) but would cause failures in any downstream analysis or visualization code that relies on string-based class name lookups.

---

## Data Cleaning

Data cleaning was performed using [`scripts/cleaning_data.py`](https://github.com/noah-y-yi/team-jn/blob/main/scripts/cleaning_data.py). The primary goal was to quantify and address the class imbalance identified during profiling. An exploratory notebook accompanying this work is located at [`scripts/notebooks/cleaning_data_pre.ipynb`](https://github.com/noah-y-yi/team-jn/blob/main/scripts/notebooks/cleaning_data_pre.ipynb).

### Operation 1: Class Weight Computation

**Issue addressed:** Severe class imbalance (69:1 annotation ratio).

**Operation:** Per-class weights were computed using the inverse-frequency formula:

```
weight(class_i) = total_annotations / (num_classes × count(class_i))
```

This formula assigns higher weights to rarer classes, reflecting the proportional attention the model should pay to each class during training. A class with half the average number of samples receives twice the average weight. The computed weights are saved to [`data/class_weights.json`](https://github.com/noah-y-yi/team-jn/blob/main/data/class_weights.json).

**Result:** Weights ranged from 0.047 (for `banana_sigatoka`, the most common class) to 3.238 (for `cauliflower_Blackrot`, the rarest). These weights serve as a reference artifact for future training runs that implement weighted loss. For example, by passing them as the `weight` parameter to PyTorch's `nn.CrossEntropyLoss`.

Note: In the current trained model, these weights were computed as a cleaning artifact but were not yet applied to the loss function during training, as the Ultralytics training pipeline does not expose a direct per-class weight parameter without a custom trainer implementation. This is a known limitation discussed further under Future Work.

### Operation 2: Class Name Normalization (Documented)

**Issue addressed:** Malformed class names with extra whitespace and inconsistent capitalization.

**Operation:** The two problematic class names were identified:

* `cauliflower_bacterial _spot _rot` → should be `cauliflower_bacterial_spot_rot`  
* `cauliflower_Blackrot` → should be `cauliflower_blackrot`

These corrections are documented here for application to the [`data.yaml`](https://github.com/noah-y-yi/team-jn/blob/main/data/data.yaml) file and any downstream processing scripts. Because YOLO training uses integer class indices rather than string names, these inconsistencies did not affect model training. Correction is recommended before any string-based downstream tooling is built.

### What Was Not Changed

No images or annotation bounding boxes were modified, removed, or resampled. The original YOLO `.txt` label files remain intact. This decision was made deliberately: the structural QA found no corrupt or invalid annotation files, and the dataset authors' train/valid/test split was preserved without modification to ensure comparability with any future work that uses the same dataset.

---

## Findings


---

## Future Work


---

## Challenges


---

## Reproducing

The following steps reproduce the full pipeline from data acquisition through model evaluation.

### Prerequisites

* Python 3.10 or later  
* A CUDA-compatible GPU (highly recommended for training; CPU is supported but slow)  
* \~5 GB of free disk space for the dataset (Acquisition steps are outlined here [data/README.md](https://github.com/noah-y-yi/team-jn/blob/main/data/README.md))

### 1\. Clone the Repository

```
git clone https://github.com/noah-y-yi/team-jn.git
cd team-jn
```

### 2\. Setup Virtual Environment

```
# Within the main directory, open a new terminal window and run this command below to create a new virtual environment for the packages

python -m venv .venv
```

### 3\. Install Dependencies

```
pip install -r requirements.txt
```

### 4\. Download the Dataset

The dataset cannot be included in the repository due to file size. Download manually from Mendeley Data:

1. Visit: [https://data.mendeley.com/datasets/6243z8r6t6/1](https://data.mendeley.com/datasets/6243z8r6t6/1)  
2. Click **Download All** to obtain `Multi-Crop Disease Dataset.zip` (\~978 MB)  
3. Extract to the `data/` directory:

```
unzip "Multi-Crop Disease Dataset.zip" -d data/
```

The expected path after extraction: `data/Multi-Crop Disease Dataset/Multicrop Disease Dataset/Multicrop Disease Dataset/`

### 5\. Run Data Profiling

```
python scripts/profile_dataset.py
```

Output: [`data/profile_results.json`](https://github.com/noah-y-yi/team-jn/blob/main/data/profile_results.json)

### 6\. Run Data Cleaning

**Note:** `scripts/cleaning_data.py` contains hardcoded absolute paths. Update lines 14 and 41 to match your local repository path before running.

```
python scripts/cleaning_data.py
```

Output: [`data/class_weights.json`](https://github.com/noah-y-yi/team-jn/blob/main/data/class_weights.json)

### 7\. Train the Model (Optional — Pre-trained Model Included)

The trained model is already included at `results/best.pt`. To retrain from scratch:

1. Open [`scripts/train_model.py`](https://github.com/noah-y-yi/team-jn/blob/main/scripts/train_model.py)  
2. Uncomment the corresponding `if __name__ == '__main__':` block  
3. Ensure `scripts/yolo26n.pt` is present (pretrained base model)  
4. Run:

```
python scripts/train_model.py
```

Training runs for 100 epochs on GPU (`device='cuda'`). Replace with `device='cpu'` if no GPU is available. Checkpoints save automatically to `runs/detect/train-*/weights/`.

### 8\. Evaluate the Model

```
python scripts/test_model.py
```

This runs the model against the test split and prints per-class metrics and overall mAP scores. Visualization outputs (confusion matrix, PR curves) save to `runs/detect/val/`.

### Automated Workflow (Snakemake)

To run steps 4–7 in sequence automatically:

```
snakemake --cores 1
```

See [`Snakefile`](https://file+.vscode-resource.vscode-cdn.net/Users/jasonwu/Documents/Team-jn/Snakefile) for the complete workflow definition. Use `--cores 4` or more to parallelize preprocessing steps.

---

## References

1. Prem Kumar E. (2024). *Multi-Crop Disease Dataset* (Version 1). Mendeley Data. [https://doi.org/10.17632/6243z8r6t6.1](https://doi.org/10.17632/6243z8r6t6.1)  
2. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (Version 8.0.0) \[Software\]. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
3. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *Proceedings of CVPR 2016*. [https://doi.org/10.1109/CVPR.2016.91](https://doi.org/10.1109/CVPR.2016.91)  
4. Bendre, A., Tupe, P., Dixit, V., & Kshirsagar, V. (2025). *Cauliflower Disease Image Dataset* (Version 1). Mendeley Data. [https://doi.org/10.17632/x26px3xnmy.1](https://doi.org/10.17632/x26px3xnmy.1)  
5. Sasmal, B., Das, A., Dhal, K. G., Saheb, B., Abu Khurma, R., & Castillo-Valdivieso, P. A. (2024). *A novel groundnut leaf dataset* (Version 2). Mendeley Data. [https://doi.org/10.17632/x6x5jkk873.2](https://doi.org/10.17632/x6x5jkk873.2)  
6. Rashid, M. R. A., Hasan, M., Gani, R., Khan Tarin, T., Kamara, R., & Rabbi, S. F. (2024). *Image Dataset for Radish Plant Leaf Disease Detection* (Version 1). Mendeley Data. [https://doi.org/10.17632/s973cz2jcd.1](https://doi.org/10.17632/s973cz2jcd.1)  
7. Sharker Nirob, M. A., Siam, A. K. M. F. K., Bishshash, P., & Assaduzzaman, M. (2025). *Chili Plant Leaf Disease and Growth Stage Dataset* (Version 1). Mendeley Data. [https://doi.org/10.17632/w9mr3vf56s.1](https://doi.org/10.17632/w9mr3vf56s.1)
