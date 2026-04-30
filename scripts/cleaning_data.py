"""
Data Cleaning and Formating script that cleans any
data quality issues in the profiling step.

"""

import re
import json



# Load profile results from profiling step
# relative path may differ from user to user
with open("/Users/jasonwu/Documents/Team-jn/data/profile_results.json", "r") as f:
    profile = json.load(f)

# retrieve data 
class_names = profile["class_names"]
class_dist = profile["combined_class_distribution"]
num_classes = profile["num_classes"]
total_annotations = profile["total_annotations"]


# Compute class weights: the higher the weight = rarer class
# Formula: total / (num_classes * class_count)

class_weights = {}
for idx, name in enumerate(class_names):
    count = class_dist.get(str(idx), 1)
    weight = total_annotations / (num_classes * count)
    class_weights[idx] = round(weight, 4)

output = {
    "class_weights": class_weights,
    "class_names": class_names
}


# write output to class_weights.json

with open("/Users/jasonwu/Documents/Team-jn/data/class_weights.json", "w") as f:
    json.dump(output, f, indent=2)