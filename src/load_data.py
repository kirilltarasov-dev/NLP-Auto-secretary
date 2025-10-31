from datasets import load_dataset
import pandas as pd
import os 
import json

os.makedirs("data/banking77", exist_ok=True)

#Downloading dataset
dataset = load_dataset("legacy-datasets/banking77")

#convert to DataFrame
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Save locally
train_df.to_csv("data/banking77/train.csv", index=False)
test_df.to_csv("data/banking77/test.csv", index=False)

# Save label names for inference
try:
    label_names = dataset["train"].features["label"].names
    with open("data/banking77/label_names.json", "w") as f:
        json.dump(label_names, f)
except Exception:
    pass

print("DataSet is saved")
 

