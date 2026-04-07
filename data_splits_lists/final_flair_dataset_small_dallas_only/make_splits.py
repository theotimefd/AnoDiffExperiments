import os
import csv
import random
# Set the random seed for reproducibility
random.seed(42)

BETTIK_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
#BETTIK_DIR = "/home/theotime/bettik/"
#BETTIK_DIR = "/home/fehrdelt/bettik/"



final_adc_dataset_dallas = "datasets/final_adc_dataset_small/Dallas_registered/"
final_flair_dataset_dallas = "datasets/final_flair_dataset_small/dallas_registered/"
final_t1_dataset_dallas = "datasets/final_t1_dataset_small/dallas_registered/"


filelist_dallas_adc = os.listdir(BETTIK_DIR+final_adc_dataset_dallas)

filelist_dallas_flair = os.listdir(BETTIK_DIR+final_flair_dataset_dallas)

filelist_dallas_t1 = os.listdir(BETTIK_DIR+final_t1_dataset_dallas)


filelist_dallas_adc_names = ['_'.join(filename.split('_')[:2]) for filename in filelist_dallas_adc]
filelist_dallas_flair_names = ['_'.join(filename.split('_')[:2]) for filename in filelist_dallas_flair]
filelist_dallas_t1_names = ['_'.join(filename.split('_')[:2]) for filename in filelist_dallas_t1]

# only keep files that have adc flair and t1 versions, i.e. the intersection of the three lists
final_filelist_names_intersection = list(set(filelist_dallas_adc_names) & set(filelist_dallas_flair_names) & set(filelist_dallas_t1_names))

filelist_dallas_flair = [final_flair_dataset_dallas + item for item in filelist_dallas_flair if '_'.join(item.split('_')[:2]) in final_filelist_names_intersection]



random.shuffle(filelist_dallas_flair)


cutoff = int(0.8 * len(filelist_dallas_flair))
train_sublist = filelist_dallas_flair[:cutoff]

val_sublist = filelist_dallas_flair[cutoff:cutoff + (len(filelist_dallas_flair)-cutoff) // 2]

test_sublist = filelist_dallas_flair[cutoff + (len(filelist_dallas_flair)-cutoff) // 2:]

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