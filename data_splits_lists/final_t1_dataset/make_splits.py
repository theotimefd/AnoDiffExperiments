import os
import csv
import random
# Set the random seed for reproducibility
random.seed(42)

BETTIK_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
#BETTIK_DIR = "/home/theotime/bettik/"
#BETTIK_DIR = "/home/fehrdelt/bettik/"

dataset_name = "final_t1_dataset"

final_t1_dataset_ixi = f"datasets/{dataset_name}/ixi_registered_final/"
final_t1_dataset_oasis = f"datasets/{dataset_name}/oasis_registered/"

filelist_ixi = os.listdir(BETTIK_DIR+final_t1_dataset_ixi)
filelist_ixi = [final_t1_dataset_ixi + item for item in filelist_ixi]

filelist_oasis = os.listdir(BETTIK_DIR+final_t1_dataset_oasis)
filelist_oasis = [final_t1_dataset_oasis + item for item in filelist_oasis]

combined_filelist = filelist_ixi + filelist_oasis
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
        excluded_files = [f"datasets/{dataset_name}/" + row[0] for row in reader]
print("Excluded files:", excluded_files)

with open("train.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in train_sublist:
        if item not in excluded_files:
            writer.writerow([item])
        else:
            print(f"Excluding {item} from train set")

with open("val.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in val_sublist:
        if item not in excluded_files:
            writer.writerow([item])
        else:
            print(f"Excluding {item} from val set")

with open("test.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in test_sublist:
        if item not in excluded_files:
            writer.writerow([item])
        else:
            print(f"Excluding {item} from test set")