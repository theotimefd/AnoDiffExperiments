import os
import csv
import random
# Set the random seed for reproducibility
random.seed(42)

#BETTIK_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
#BETTIK_DIR = "/home/theotime/bettik/"
BETTIK_DIR = "/home/fehrdelt/bettik/"

#dallas_dir = "datasets/Dallas_T2_FLAIR_extracted_brain_registered/"
dallas_dir = "datasets/Dallas_T2_FLAIR_extracted_brain_registered_rotated/"
#hcp_ya_dir = "datasets/ADC_Human_Connectome_Project_Young_Adult_HCP-YA_resized128/"

filelist_dallas = os.listdir(BETTIK_DIR+dallas_dir)
filelist_dallas = [dallas_dir + item for item in filelist_dallas]

#filelist_hcp_ya = os.listdir(BETTIK_DIR+hcp_ya_dir)
#filelist_hcp_ya = [hcp_ya_dir + item for item in filelist_hcp_ya]

#combined_filelist = filelist_dallas + filelist_hcp_ya
random.shuffle(filelist_dallas)


cutoff = int(0.8 * len(filelist_dallas))
train_sublist = filelist_dallas[:cutoff]

val_sublist = filelist_dallas[cutoff:cutoff + (len(filelist_dallas)-cutoff) // 2]

test_sublist = filelist_dallas[cutoff + (len(filelist_dallas)-cutoff) // 2:]

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
