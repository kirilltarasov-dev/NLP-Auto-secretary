from datasets import load_dataset
import pandas as pd
import os 

os.makedirs("data/banking77", exist_ok=True)

#Downloading dataset
dataset = load_dataset("legacy-datasets/banking77")

#convert to DataFrame
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])


# Save locally
train_df.to_csv("data/banking77/train.csv", index=False)
test_df.to_csv("data/banking77/test.csv", index=False)

print("DataSet is saved")

