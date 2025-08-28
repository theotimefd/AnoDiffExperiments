import os
import csv
import random
# Set the random seed for reproducibility
random.seed(42)

#BETTIK_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
#BETTIK_DIR = "/home/theotime/bettik/"
BETTIK_DIR = "/home/fehrdelt/bettik/"


final_flair_dataset_dallas = "datasets/final_flair_dataset_small/dallas_registered/"
final_flair_dataset_lemon = "datasets/final_flair_dataset_small/lemon_registered/"

filelist_dallas = os.listdir(BETTIK_DIR+final_flair_dataset_dallas)
filelist_dallas = [final_flair_dataset_dallas + item for item in filelist_dallas]

filelist_lemon = os.listdir(BETTIK_DIR+final_flair_dataset_lemon)
filelist_lemon = [final_flair_dataset_lemon + item for item in filelist_lemon]

combined_filelist = filelist_dallas + filelist_lemon
random.shuffle(combined_filelist)


cutoff = int(0.8 * len(combined_filelist))
train_sublist = combined_filelist[:cutoff]

val_sublist = combined_filelist[cutoff:cutoff + (len(combined_filelist)-cutoff) // 2]

test_sublist = combined_filelist[cutoff + (len(combined_filelist)-cutoff) // 2:]

# files contained in exclude.csv will not be included in the splits
EXCLUDE_FILES = True
excluded_files = []
exclude_file_path = "exclude.csv"

if EXCLUDE_FILES and os.path.exists(exclude_file_path):
    with open(exclude_file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        excluded_files = [row[0] for row in reader]
print("Excluded files:", excluded_files)

with open("train.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in train_sublist:
        if item not in excluded_files:
            writer.writerow([item])

with open("val.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in val_sublist:
        if item not in excluded_files:
            writer.writerow([item])

with open("test.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in test_sublist:
        if item not in excluded_files:
            writer.writerow([item])
