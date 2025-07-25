import os
import csv

BETTIK_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
#BETTIK_DIR = "/home/theotime/bettik/"

dataset_dir = BETTIK_DIR+"datasets/Dallas_Computed_ADC_extracted_brain_registered_rotated_padded/"

filelist = os.listdir(dataset_dir)
train_sublist = filelist[:763]
val_sublist = filelist[763:]
test_sublist = filelist[763:]

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