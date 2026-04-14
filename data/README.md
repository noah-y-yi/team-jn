# Dataset Documentation

## Primary Dataset: Multi-Crop Disease Dataset

**Source:** [Mendeley Data — Multi-Crop Disease Dataset](https://data.mendeley.com/datasets/6243z8r6t6/1)  
**License:** CC BY 4.0  
**Publisher:** Prem Kumar E, VIT Chennai  
**Version:** 1  

### Description

A comprehensive collection of annotated images of diseased and healthy leaves across five agricultural crops. Leaf samples were collected from real farming fields across Chengalpattu, Kanchipuram, and Krishnagiri districts in Tamil Nadu, India (November–January 2024), captured using high-resolution digital cameras and 200 MP mobile phone cameras.

- **Total images:** 23,000+
- **Annotation format:** YOLO (convertible to COCO/VOC)
- **Crops covered:** Banana, Chilli, Radish, Groundnut, Cauliflower

### Disease Classes per Crop

| Crop | Disease Examples |
|---|---|
| Banana | Sigatoka, Anthracnose |
| Chilli | Anthracnose, Bacterial Spot |
| Radish | Black Leaf Spot, Downy Mildew |
| Groundnut | Rust, Early Leaf Spot |
| Cauliflower | Downy Mildew, Black Rot |

Each crop folder contains both healthy and diseased sample images.

---

## How to Acquire the Dataset

The dataset file (`Multi-Crop Disease Dataset.zip`, ~978 MB) is **not tracked by Git** due to GitHub's file size limit. To reproduce this project locally:

1. Download the dataset from [https://data.mendeley.com/datasets/6243z8r6t6/1](https://data.mendeley.com/datasets/6243z8r6t6/1)
2. Place the downloaded zip file in the root of the repository
3. Extract to the `data/` directory:

```bash
unzip "Multi-Crop Disease Dataset.zip" -d data/
```

---

## Expected Directory Structure After Extraction

```
data/
├── README.md                  ← this file
├── Banana/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── Chilli/
│   ├── images/
│   └── labels/
├── Radish/
│   ├── images/
│   └── labels/
├── Groundnut/
│   ├── images/
│   └── labels/
└── Cauliflower/
    ├── images/
    └── labels/
```

> Note: Exact subdirectory naming may vary slightly from the source zip. See [scripts/profile_dataset.py](../scripts/profile_dataset.py) to automatically inspect the extracted structure.

---

## Storage Strategy

- **Raw data:** Stored locally only; excluded from version control via `.gitignore`
- **Annotations:** YOLO `.txt` format — one label file per image
- **Version control:** Only scripts, documentation, and notebooks are committed to GitHub
- **Reproducibility:** Any team member can re-acquire the data using the download link above

---

## Secondary / Test Datasets (Planned)

These external datasets will be used to evaluate model generalization after training:

| Crop | Dataset | Source |
|---|---|---|
| Banana | Black Sigatoka | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LQUWXW) |
| Chilli | Bacterial Spot | [Mendeley](https://data.mendeley.com/datasets/w9mr3vf56s/1) |
| Radish | Black Leaf Spot | [Mendeley](https://data.mendeley.com/datasets/s973cz2jcd/1) |
| Groundnut | Rust | [Mendeley](https://data.mendeley.com/datasets/x6x5jkk873/2) |
| Cauliflower | Downy Mildew | [Mendeley](https://data.mendeley.com/datasets/x26px3xnmy/1) |
