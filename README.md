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

The trained model was evaluated on the test set included in the dataset (2191 images). It utilized the Ultralytics validation pipeline ([`scripts/test_model.py`](https://github.com/noah-y-yi/team-jn/blob/main/scripts/test_model.py)). Inference ran at 4.0 ms per image on GPU (1.4 ms preprocessing, 0.2 ms postprocessing).

### Overall Performance

| Metric | Value |
| :---- | :---- |
| mAP50 | 0.714 |
| mAP50-95 | 0.515 |
| Precision | 0.739 |
| Recall | 0.670 |

The gap between mAP50 and mAP50-95 indicates that the model does well identifying the disease presence, but does worse producing bounding boxes. For the purposes of our project, we mostly care about disease identification accuracy, but for potential applications of our model, bounding boxes are important as well.

### Per-Class Performance (Sample of 15 Classes)

| Class | mAP50 | mAP50-95 | P | R |
| :---- | :---- | :---- | :---- | :---- |
| groundnut\_early\_rust | 0.995 | 0.840 | 0.744 | 0.925 |
| cauliflower\_healthy | 0.995 | 0.959 | 0.966 | 1 |
| banana\_healthy | 0.995 | 0.827 | 0.979 | 0.991 |
| groundnut\_healthy | 0.995 | 0.942 | 0.98 | 1 |
| banana\_bract\_mosaic\_virus | 0.974 | 0.894 | 0.88 | 1 |
| cauliflower\_Blackrot | 0.947 | 0.925 | 0.853 | 0.889 |
| cauliflower\_bacterial\_spot\_rot | 0.985 | 0.441 | 0.908 | 0.938 |
| groundnut\_early\_rust | 0.945 | 0.638 | 0.744 | 0.925 |
| groundnut\_late\_leaf\_spot | 0.937 | 0.724 | 0.918 | 0.856 |
| radish\_black\_leaf\_spot | 0.971 | 0.659 | 0.856 | 1 |
| banana\_sigatoka | 0.474 | 0.202 | 0.447 | 0.515 |
| groundnut\_early\_leaf\_spot | 0.216 | 0.071 | 0.509 | 0.122 |
| chilli\_whitefly | 0.177 | 0.066 | 0.521 | 0.129 |
| chilli\_leafcurl | 0.113 | 0.063 | 0 | 0 |
| chilli\_anthracnose | 0.030 | 0.011 | 0.148 | 0.0112 |

From these results, we can clearly see where performance was successful and which classes the model struggled identifying. The model has excellent performance with the healthy plants, as seen with `cauliflower_healthy`, `banana_healthy`, and `groundnut_healthy` scoring high in the mAP50-95. It is also excellent when there are visually distinctive marks, as seen with `cauliflower_Blackrot` and the darker spots.

Where the model struggles is with the chilli diseases. As seen visually, multiple chilli diseases (anthracnose, leafcurl, leafspot, whitefly) produce lesion patterns that overlap visually. The confusion matrix (see [`results/confusion_matrix.png`](https://github.com/noah-y-yi/team-jn/blob/main/results/confusion_matrix.png)) shows substantial cross-class confusion within this group, supporting our visual analysis. A significant outlier in performance is the `groundnut_early_leaf_spot` class, which despite having the largest number of images/annotations (6323), had a low score. This indicates that just having a great number of annotations does not guarantee high detection accuracy. There are additional factors such as visual ambiguity that might throw off models.

Visualization artifacts such as PR curves, F1 curves, confusion matrices, and validation batch predictions are stored in the [`results/`directory](https://github.com/noah-y-yi/team-jn/tree/main/results).

---

## Future Work

There were several lessons learned that pave the way for future work:

**1\. Apply class weights to training loss.** The class weights computed in [`data/class_weights.json`](https://github.com/noah-y-yi/team-jn/blob/main/data/class_weights.json) were not applied during training due to Ultralytics' training API limitations. A custom Ultralytics trainer subclass could override the default loss function to incorporate these weights. This would directly penalize misclassification of rare classes more heavily and is expected to improve recall for the lowest-performing Chilli and Cauliflower classes. Documentation to implement a custom trainer with class weights can be found [here](https://docs.ultralytics.com/guides/custom-trainer/#adding-class-weights).

**2\. Data augmentation on smaller class sets.** For classes with much fewer (\<300) training annotations, augmentation (flips, rotations, color filters, mosaics) can increase the training size. Ultralytics already contains features to implement data augmentation natively. By applying it to the underrepresented classes, it can possibly improve targeted accuracy. Documentation to implement data augmentation using Ultralytics can be found [here](https://docs.ultralytics.com/guides/yolo-data-augmentation/).

**3\. Scale up the base pretrained model.** In this project, we used the YOLO26n pretrained model. Also available but significantly heavier are the YOLO26s, YOLO26m, YOLO26l, and YOLO26x models. By upgrading the model, it may improve performance on classes where we deal with visual ambiguity, like the chilli class, and this would not require us to modify any training data. Support for specific YOLO models is discussed briefly [here](https://docs.ultralytics.com/models/yolo26/#what-tasks-does-yolo26-support).

**4\. Testing with external datasets.** In our status report (and `data/README.md`), we identified five external datasets that we were planning to use as more generalized testing. This is because they did not come from the same dataset we used to train and validate our model. By testing the model against images from different cameras, regions, and lighting conditions, it would reveal if our model had learned generalized features or was overfit with the training data. This would be one of the most obvious steps before launching as a product.

**5\. Two-step crop classification.** Because the current model sees the 30 classes equally without first identifying the crop, it could be potentially causing confusion with classes such as the chillis. By first identifying the crop, then the disease, it could solve this confusion and increase performance. It could also lead to higher potential external uses of the model for deployment.

**6\. Deployment.** We purposely export the model as an ONNX format ([`results/best.onnx`](https://github.com/noah-y-yi/team-jn/blob/main/results/best.onnx)) for future deployment. If deployed on a device with a camera to be used in the real world, it could lead to further research and application. Article on deploying an ONNX model can be read [here](https://medium.com/tr-labs-ml-engineering-blog/model-deployment-with-onnx-7b45b82da71c).

**Specific lessons learned:** Class imbalance can potentially be fixed by implementing some more customization while training the model. It requires non-trivial Ultralytics usage which we did not have the time to explore. For future projects, given this baseline model experiment, it’s possible to produce much better results and application. Future work should consider these in advance before training rather than saving it for after. Additionally, some of the diseases for specific crops are just challenging to identify in the first place. By having a greater amount of high-quality annotations from domain experts, it’s possible to increase reliability during training and can also produce better results.

---

## Challenges

There were some difficult challenges that we encountered while completing this work:

**1\. Large dataset size and storage constraints.** GitHub enforces a hard 100 MB file size limit, which meant our dataset (around 978 MB zip file) was unable to to be committed. In order to get around this challenge, each of us downloaded the dataset (specifically the zip file) locally, and the data was excluded via the [`.gitignore`](https://github.com/noah-y-yi/team-jn/blob/main/.gitignore) file. Specific instructions on how to reproduce and set up the dataset locally is in the [`data/README.md`](https://github.com/noah-y-yi/team-jn/blob/main/data/README.md) file, which also includes the dataset source URL and verification steps.

**2\. Class imbalance and its negative impact on training.** After profiling, we identified a huge class imbalance (69:1 ratio) between the most and least represented classes. While we computed custom class weights to address this, we were unable to successfully integrate them into the Ultralytics training pipeline due to complexity and time constraints. The model was then trained without the custom weighting, and the results reflect it, with lowly annotated classes performing the poorest.

**3\. Compute resources for training.** Training a YOLO model on 15,310 images for 100 epochs is computationally expensive. Running on a CPU would have been physically incredibly slow. Thus, we relied on local GPU hardware. Training was run on a Dell XPS 16 9640 with a built-in NVIDIA GeForce RTX 4050 Laptop GPU. Without a local GPU, users would likely have to rely on cloud resources such as Google Colab or utilizing NCSA resources. Using Google Colab would introduce session timeout risks during long training runs, which we account for by including the resume block in `train_model.py`.

**4\. Setting up CUDA to run PyTorch with a local GPU.** One of the most frustrating parts of the training process was to set up NVIDIA’s CUDA software to run PyTorch with our local GPU. There were some prerequisites such as [downloading Visual Studio](https://visualstudio.microsoft.com/downloads/) with the “Desktop development with C++” workload, installing the most recent [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit), ensuring the device environment variables were correctly set up, and ensuring that `nvcc –version` returns correctly. Because we installed the ultralytics package in our virtual environment already, we then had to uninstall the included PyTorch library and [reinstall the package](https://pytorch.org/get-started/locally/) with our CUDA version. The difficult part of this challenge was just finding working versions and determining the correct steps for installing.

**5\. Poor chilli class performances.** Several disease classes within the chilli crop produce lesion patterns that are difficult to distinguish even for human reviewers. This is reflected in the testing results where we can see that `chilli_anthracnose`, `chilli_leafcurl`, and `chilli_leafspot` all scored below 0.31 in mAP50. This challenge cannot be addressed through data cleaning or model scaling, but rather more distinctive annotation criteria or better review of ambiguous labels.

**6\. Project timeline.** The status report deadline extension incidentally pushed/compressed several of our planned tasks later in the timeline. This sped up model development, testing, and documentation phases into a shorter window. As a result, external dataset acquisition and testing was not completed within the timeframe.

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

(OPTIONAL) Setup PyTorch with CUDA:

1. Visit: https://developer.nvidia.com/cuda/toolkit  
2. Click **Download Now** and ensure it’s installed correctly  
3. Run these commands

Uninstall the PyTorch from Ultralytics

```
py -m pip uninstall torch torchvision torchaudio -y
```

Install PyTorch with the CUDA version

```
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 # REPLACE WITH CORRECT URL
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
