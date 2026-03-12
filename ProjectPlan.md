# Team jn Project Plan

# Overview:

The goal of our project is to develop a computer vision based system capable of identifying crop diseases from plant images. The use of early detection technology for plant diseases is critical for improving crop yield, reducing agricultural losses, and supporting farming practices. Manual disease identification is a time-consuming task that requires expertise, so the project aims to explore how machine learning and image analysis can assist in the automated disease detection.

Our approach for this project will involve the collection and integration of multiple agricultural datasets, including the Medeley’s “Crop Pest and Disease Detection” dataset, which contains labeled images of crops with different disease conditions. We will combine the image data with supplementary metadata where we will then use it to support analysis and model training.

Our project will consist of several phases. First, we will acquire and explore the datasets, examining the structure, labels, and quality. Next, we will perform any data cleaning, preprocessing, deidentification, and integration to prepare the data for analysis. We will then train and evaluate our computer vision models capability to detect disease patterns in the crop images. Potential methods we plan to do include convolutional neural networks (CNNs) and transfer learning using pretrained image classification models.

Lastly, we will evaluate the model’s performance using metrics like accuracy, precision, recall, and F1-score, and confusion matrix to analyze which diseases and crops are most reliably detected from our classifications. The results will demonstrate the use of machine learning tools to support agricultural monitoring and decision making.

# Team:

The team comprises two members Noah and Jason. Noah is a Junior in Engineering and Jason is a Senior in the iSchool. Jason and Noah will be working collaboratively by communicating via social media and zoom meetings to get assignments and milestones completed. Since both are able to work with data, work will be split evenly between Jason and Noah where we will consult with each other before taking on tasks that need to be completed. For example, for this Project Plan, we both consulted each other on what sections need to be completed and we were able to work together and come up with a plan to get this done.

# Research/Business Question:

This project aims to answer the following analytical questions:

1. **Can computer vision models accurately detect crop diseases from plant images?**

2. **Which crop diseases are most reliably identified by machine learning models?**

3. **How does model performance vary across different crop types and disease categories?**

4. **What preprocessing or feature extraction techniques improve disease detection accuracy?**

5. **How could automated disease detection systems assist farmers or agricultural monitoring systems?**

These questions will help evaluate the feasibility of using image-based machine learning systems for agricultural disease detection and decision support.

# Datasets:

[Multi-Crop Disease Dataset](https://data.mendeley.com/datasets/6243z8r6t6/1) (CC BY 4.0), ALTERNATIVELY, [Dataset for Crop Pest and Disease Detection](https://data.mendeley.com/datasets/bwh3zbpkpv/1) (CC BY 4.0)  
The *Multi-Crop Disease Dataset* was curated by VIT Chennai’s Prem Kumar E. This dataset presents a comprehensive collection of annotated images of diseased and healthy leaves across five important agricultural crops: Banana, Chilli, Radish, Groundnut, and Cauliflower. The dataset was created to support research in plant disease detection, precision agriculture, and deep learning-based crop monitoring systems.

Leaf samples were collected from real farming fields across Chengalpattu, Kanchipuram, and Krishnagiri districts in Tamil Nadu, India, between November and January 2024, captured using high-resolution digital cameras and 200 MP mobile phone cameras. Data contains over 23,000 images labeled using bounding box annotations, including Banana, Chilli, Radish, Groundnut, and Cauliflower. Healthy and diseased samples were captured across multiple disease types (e.g., Sigatoka, Anthracnose, Rust, Downy Mildew, etc.). 

According to the publisher, images are organized by crop name and disease class and annotations are provided in YOLO format (can be converted to COCO/VOC). Meaning this dataset is suitable for training CNN, YOLO, Faster R-CNN, or ViT models for plant disease classification and localization.

Some of the potential applications for this dataset include:

* Real-time disease diagnosis in smart farming systems  
* Academic research in plant pathology and computer vision  
* Benchmarking object detection models in agricultural settings

In our case, we would like to use this dataset to train a model. There are two straightforward ways we can do this with this dataset. The first way is to finetune a pretrained model with the given dataset. The second one is to train a model from scratch. Either method is easily accomplished using the open source [ultralytics](https://www.ultralytics.com/) object detection model, which the dataset is already formatted in.

If we determine that this pipeline does not fit the requirements of the project, then we can pivot to using the *Dataset for Crop Pest and Disease Detection*. This dataset presents crop pests/disease datasets sourced from local farms in Ghana. The raw images number over 24,000 images, and this dataset also includes already augmented images that increase the count to over 100,000. This dataset is suitable for both fine tuning and training a model from scratch without bounding box annotations. The data is also not preformatted for a specific model, so we will have more flexibility there.

Choosing either of these datasets allows us to answer our research questions.

After training a model, it might be a good idea to test it on brand new, unseen data from a different source. So to complement the model training data, we also are looking for images from the same crops to test.

Here are some examples of potential test data for:

* Banana \- [Black Sigatoka](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LQUWXW):  
  * The banana images dataset was created to contribute to the study of banana diseases diagnostics. The images target the diagnostics of Black Sigatoka and Fusarium Wilt Race 1 diseases.  
* Chilli \- [Bacterial Spot](https://data.mendeley.com/datasets/w9mr3vf56s/1):  
  * This dataset is a comprehensive resource for researchers and professionals in agriculture, machine learning, and computer vision, focusing on chili plant disease detection and growth stage classification.  
* Radish \- [Black Leaf Spot](https://data.mendeley.com/datasets/s973cz2jcd/1):  
  * Images of radish leaves for disease detection from Bangladesh.  
* Groundnut \- [Rust](https://data.mendeley.com/datasets/x6x5jkk873/2):  
  * The image of the groundnut leaves was collected from the agricultural land of Ramchandrapur village in Purba Medinipur district of West Bengal, India. Contains over 1,700 images in JPEG format.  
* Cauliflower \- [Downy Mildew](https://data.mendeley.com/datasets/x26px3xnmy/1):  
  * The dataset comprises images of cauliflower plants classified based on the various diseases attacking them. The dataset is meant for applications in plant disease classification and prediction methods. It is aimed at helping plant pathologists and farmers to identify the diseases affecting cauliflower at an early stage, thus preventing their occurrence.

# Timeline:

| Date/Week | Task | Description | Responsible |
| :---: | :---: | :---: | :---: |
| Feb 17 \- Completed | Team Formation | Form team, Create Github repo, assign roles, setup folders | All  |
| Mar 1 \- Mar 10 | Project Plan | Define project scope, research questions, datasets, and methodology. Create [ProjectPlan.md](http://ProjectPlan.md) and submit Milestone 2 on Github | All |
| Mar 11 \- Mar 17 | Data Acquisition | Download and document datasets and the secondary dataset. Write scripts or documentation for data acquisition and verify data integrity. | All |
| Mar 18 \- Mar 23 | Data Storage & Organization | Organize datasets using a structured directory format. Define file formats, naming conventions, and storage strategy. Prepare datasets for integration. | All |
| Mar 24 \- Mar 30 | Data Profiling & QA | Evaluate dataset characteristics including missing values, label consistency, and class distribution. Document data quality issues and initial findings. | All |
| Mar 31 | Milestone 3: Interim Status Report | Submit [StatusReport.md](http://StatusReport.md) summarizing progress, challenges, updated timeline, and contributions | All |
| Apr 1 \- Apr 7 | Data Cleaning | Apply cleaning operations such as correcting labels, handling missing values, removing corrupted images, and standardizing formats. Document all cleaning operations. | All |
| Apr 8 \- Apr 14 | Data Integration | Integrate datasets by aligning crop types, disease labels, or metadata attributes. Develop scripts in Python/Pandas or SQL to combine datasets. | All |
| Apr 15 \- Apr 20 | Model Development | Develop computer vision models for crop disease detection using image classification techniques such as convolutional neural networks (CNNs) or transfer learning. | All |
| Apr 21 \- Apr 24 | Analysis & Visualization | Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score. Create visualizations of results and dataset characteristics. | All |
| Apr 25 \- 27 | Workflow Automation | Implement an automated workflow (e.g., Snakemake or scripts) to reproduce the pipeline from data acquisition to results. | All |
| Apr 25 \- Apr 30 | Documentation & Reproducibility | Prepare project documentation including metadata, data dictionary, workflow instructions, and dependency specifications (`requirements.txt`). | All |
| May 1 \- May 2 | Final Report Preparation | Complete the [README.md](http://README.md) project report including summary, data profile, data quality assessment, cleaning documentation, findings, and challenges. | All |
| May 3 | Final Project Submission | Create final GitHub release containing report, datasets, scripts, workflow, and documentation. Submit final project URL. | All |

	

# Constraints:

Originally, this project was supposed to make use of two datasets. One would be a dataset of images to train a model visually. The second would be a complementary dataset full of rich metadata to support the predictive power of a potential combined model. The main constraint with this idea is that none of the image datasets contained the metadata necessary to join the two models. Meaning, there was not enough information on the coordinates of where each photo was taken, and no specific time or date of when it was taken.

Another constraint on this project is training time. If we were to train a model from scratch, it would take an exponentially longer time than fine-tuning a model. Because we would like to explore different methods in this project, we will likely build both types of models. The big problem is that it is a time consuming task and we will need to plan around the timing.

# Gaps:

A gap that this current project plan contains is which hardware we will be using for this project. We will need to determine whether to use a local GPU or a cloud-based GPU (e.g. Google Colab). If we were only using a CPU, training a model from scratch would be highly inefficient and create a bottleneck.

Another gap is how we are storing the data. There are a lot of images, and we will need to determine how they will be stored. We can either store the data locally to potentially speed up training, or we can stream from the cloud.

Depending on the dataset we are moving forward with, we can either preprocess the images with the annotated bounding boxes to crop them before sending through a classifier, or we can train an object detection model. This also begs the question of which type of model we should use (classifier vs. object detection).
