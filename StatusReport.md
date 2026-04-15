**Interim Status Report \- Team jn**  
**Project:** Crop Disease Detection Using Computer Vision  
**Report Date:** Apr 14, 2026  
**Team Members**: Noah Yi, Jason Wu

**1\. Task Updates:**  
**———————————————————————————————————————**  
**Team Formation \- Completed (Feb 17\)**  
We formed the team on Feb 17, 2026\. We both talked about what we wanted to do with the project and any ideas that we both had. We established communication via social media, Zoom, created the shared GitHub repo, and established roles. All of the setup for this project was completed that day.

**Project Plan \- Completed \- (Mar 1 to 10\)**  
The project plan was completed and committed to the repository as [ProjectPlan.md](https://github.com/noah-y-yi/team-jn/blob/main/ProjectPlan.md). The plan outlined the project overview, team structure, research questions, dataset selection rationale, project timeline, constraints, and any known gaps we found. We both contributed to drafting and revising the document before submitting. Additionally we consulted Professor Carboni on whether our project was feasible as we had trouble finding an additional dataset to integrate with. However Professor Carboni, found our project interesting and suggested we just split our one dataset into two separate data sets and perform the integration via that method.

Feedback from Milestone 2 didn’t really apply in our case because Professor Carboni said it was fine for us to utilize just our 1 massive dataset and perform the training and testing via that one method.

**Data Acquisition \- Completed (Apr 13\)**

The primary dataset is the [Multi-Crop Disease Dataset](https://data.mendeley.com/datasets/6243z8r6t6/1) (CC BY 4.0) from Mendeley Data which was successfully downloaded onto our local machines. The dataset contains 21,895 images and 42,667 annotations of the diseased and healthy plant leaves across five crops: Banana, Chill, Radish, Groundnut, and Cauliflower. The images are annotated in YOLO format with bounding boxes for both healthy and diseased samples.

However due to GitHub’s limitation of 100 MB file size limits, the raw dataset zip (\~978 MB) is excluded from our version control via the .gitignore. The dataset download instruction and source links are documented in a separate folder within our repo in [data/README.md](https://github.com/noah-y-yi/team-jn/blob/main/data/README.md), providing detailed instructions for reproducing the data and acquisition step we completed.

**Data Storage & Organization \- Completed (Apr 13\)**  
Along the successful completion of the data acquisition step, The dataset was organized by crop and split into training and validation subsets following the YOLO directory convention. The structure after extraction is documented in the [data/README.md](https://github.com/noah-y-yi/team-jn/blob/main/data/README.md) as well.

Storage decisions made:

* Raw data is stored locally only and excluded from Git via .gitignore  
* Only scripts, documentation, and notebooks are committed to the repository  
* YOLO .txt annotation format is retained from the source dataset  
* Secondary test datasets (per-crop external sources) are identified and documented in [data/README.md](https://github.com/noah-y-yi/team-jn/blob/main/data/README.md) with source URLs for future acquisition

**Data Profiling \+ Quality Inspection \- Completed (Apr 13\)**   
After storage, we created a data profiling and quality assurance script written at [scripts/profile\_dataset.py](https://file+.vscode-resource.vscode-cdn.net/Users/jasonwu/Documents/Team-jn/scripts/profile_dataset.py) and executed against the extracted dataset. And the full results are saved to a json format file in our repository [data/profile\_results.json](https://file+.vscode-resource.vscode-cdn.net/Users/jasonwu/Documents/Team-jn/data/profile_results.json). The script checks each split for image/label pairings, empty files, class distribution, and annotation counts

Dataset Breakdown:

| Split | Images | Annotations |
| ----- | ----- | ----- |
| Train | 15,310 | 30,481 |
| Valid | 4,374 | 7,881 |
| Test | 2,191 | 4,305 |
| **Total** | **21,875** | **42,667** |

The train/valid/test ration is \~70%/\~20%/\~10%, which is considered a standard and normal amount for detection training. 

QA Results: When taking a look at the quality. There were no issues detected as every image file had a corresponding YOLO label file, no zero-byte or corrupt images were found, and all 30 class IDs are represented across all three splits (train/valid/test). 

Although there was no issues, one potential issue could be the class imbalance identified within the 30 disease classes. The most annotated class (banana\_sigtoka) has 6,692 annotations while the least annotated class (cauliflower\_Blackrot) has only 97 which is a ratio of \~69:1. This imbalance will be addressed later on during our data cleaning and model training phase.

**Data Cleaning \- Pending (Planned: Apr 15-21)**  
Not yet started. Will follow data profiling results and some of the planned operations include removing corrupt or zero-byte images, correcting inconsistent labels, standardizing file naming, and lastly documenting all our cleaning procedures to make it transparent to the steps we applied.

**External Dataset Testing \- Pending (Planned: Apr 22-26)**  
Not yet started. Five secondary crop-specific datasets have been identified (documented in [data/README.md](https://github.com/noah-y-yi/team-jn/blob/main/data/README.md)). We will acquire these datasets and use them to evaluate our model’s generalization.

**Model Development \- Pending (Planned: Apr 22-28)**  
Not yet started. We plan to approach this by fine-tuning a pretrained model (likely YOLOv8 via the Ultralytics library) on the primary dataset, while training a potential model from scratch for comparisons.

**Analysis & Visualization \- Pending (Planned: Apr 26-30)**  
Not yet started. Will evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix. Visualizations of class distributions and detection results will be produced.

**Workflow Automation & Documentation \- Pending (Planned: Apr 28-May 2\)**  
Not yet started.

**Final Report & Submission \- Pending (Planned: May 3-7)**  
Not yet started.

**2\. Timeline Updates:**  
**———————————————————————————————————————**

| Date | Task | Status |
| :---- | :---- | :---- |
| Feb 17 | Team Formation | Completed |
| Mar 1–10 | Project Plan (Milestone 2\) | Completed |
| Mar 11 – Apr 13 | Data Acquisition | Completed |
| Apr 13 | Data Storage & Organization | Completed |
| Apr 13–14 | Data Profiling & QA | Completed |
| Apr 15–21 | Data Cleaning | Pending |
| Apr 22–26 | External Dataset Acquisition & Testing Setup | Pending |
| Apr 22–28 | Model Development | Pending |
| Apr 26–30 | Analysis & Visualization | Pending |
| Apr 28 – May 2 | Workflow Automation & Documentation | Pending |
| May 3–5 | Final Report Preparation | Pending |
| May 7 | Final Project Submission | Pending |

**3\. Project Plan Updates:**  
**———————————————————————————————————————**

The original plan for the submission of this status report was to have a deadline of 3/31. However, due to popular request, it was extended to be due 4/14. As a result, there are many parts of our timeline that were deferred to be completed alongside this report, such as data storage, organization, and profiling, which have already been committed to the repository.

Another slight change we made to the original project plan was by keeping the directory storage with all of the labeled images, but by only storing them locally. This is due to GitHub’s file storage limits. Instructions to download them correctly can be found in data/README.md. Additionally, we are also currently evaluating whether to set up local hardware for training or utilizing Google Colab’s free resources. We plan to have this determined before model development.

A question from our project plan was determining whether to use a classifier model or object detection model for this project. Since the primary dataset is already annotated in YOLO format, we will be moving forward with object detection for our model for ease of use.

**4\. Challenges:**  
**———————————————————————————————————————**

1. **Large Dataset Size and Storage**  
Problem: GitHub has a 100 MB file limit, and the annotated image dataset we downloaded from Mendeley had a size of 978 MB, exceeding the limit.  
Solution: Each of us downloaded the dataset locally, and ignore pushing the files in our .gitignore. Instructions on how to set up the project with the data is in the data/README.md.
2. **Extended Deadline and Delayed Start**  
Problem: Due to the status report extension, we procrastinated and started work late.  
Solution: Work was completed but also rushed, and the timeline has been adjusted accordingly.
3. **Dataset Class Imbalance**  
Problem: After profiling our training set, we noticed an imbalance across the disease categories. The class banana_sigatoka has 6692 annotations while the class cauliflower_Blackrot only has 97. There were other classes that fell below 200 annotations as well. This can lead to bias towards the dominant classes.  
Solution: During the data cleaning phase, we will look at different strategies such as oversampling.
4. **Training Time**  
Problem: Training a computer vision model on a dataset of this size is expensive. Running on a local CPU would take forever.  
Solution: Depending on our local devices, we can take advantage of hardware such as GPUs. If this proves to be insufficient, it’s possible to look into Google Colab’s free GPUs and TPUs.


**5\. Individual Contributions:**  
**———————————————————————————————————————**  
**Jason Wu:**  
Worked on sections 1 and 2 of the Status Report. Also responsible for downloading and verifying the training dataset (Multi-Crop Disease Dataset), and implementing the Python scripts for profiling and QA.

**Noah Yi:**
Worked on sections 3 and 4 of the Status Report. Also involved with the research of potential solutions to challenges presented in the project such as the class imbalance and training time problems. 
