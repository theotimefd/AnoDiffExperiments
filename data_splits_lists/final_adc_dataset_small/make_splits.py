import os
import csv
import random
# Set the random seed for reproducibility
random.seed(42)

#BETTIK_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
#BETTIK_DIR = "/home/theotime/bettik/"
BETTIK_DIR = "/home/fehrdelt/bettik/"


Final_ADC_Dataset_AINI_stroke_ait = "datasets/final_adc_dataset_small/AIT_final_registered/"
Final_ADC_Dataset_Dallas = "datasets/final_adc_dataset_small/Dallas_registered/"
Final_ADC_Dataset_HCP_YA = "datasets/final_adc_dataset_small/HCP-YA_registered/"
Final_ADC_Dataset_IXI = "datasets/final_adc_dataset_small/ixi_registered/"

filelist_aini_stroke_adc = os.listdir(BETTIK_DIR+Final_ADC_Dataset_AINI_stroke_ait)
filelist_aini_stroke_adc = [Final_ADC_Dataset_AINI_stroke_ait + item for item in filelist_aini_stroke_adc]

filelist_dallas = os.listdir(BETTIK_DIR+Final_ADC_Dataset_Dallas)
filelist_dallas = [Final_ADC_Dataset_Dallas + item for item in filelist_dallas]

filelist_hcp_ya = os.listdir(BETTIK_DIR+Final_ADC_Dataset_HCP_YA)
filelist_hcp_ya = [Final_ADC_Dataset_HCP_YA + item for item in filelist_hcp_ya]

filelist_ixi = os.listdir(BETTIK_DIR+Final_ADC_Dataset_IXI)
filelist_ixi = [Final_ADC_Dataset_IXI + item for item in filelist_ixi]

combined_filelist = filelist_aini_stroke_adc + filelist_dallas + filelist_hcp_ya # + filelist_ixi
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
        excluded_files = ["datasets/final_adc_dataset_small/" + row[0] for row in reader]
#print("Excluded files:", excluded_files)

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
