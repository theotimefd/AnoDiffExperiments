import os
import csv

BETTIK_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
dataset_dir = BETTIK_DIR+"datasets/ADC_Human_Connectome_Project_Young_Adult_HCP-YA_resized128/"

filelist = os.listdir(dataset_dir)
train_sublist = filelist[:812]
val_sublist = filelist[812:]
test_sublist = filelist[812:]

with open("train.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in train_sublist:
        writer.writerow([dataset_dir+item])

with open("val.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in val_sublist:
        writer.writerow([dataset_dir+item])

with open("test.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in test_sublist:
        writer.writerow([dataset_dir+item])