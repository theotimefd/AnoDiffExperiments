import os
import csv
import random
# Set the random seed for reproducibility
random.seed(42)

BETTIK_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
#BETTIK_DIR = "/home/theotime/bettik/"
#BETTIK_DIR = "/home/fehrdelt/bettik/"

ixi_dir = "datasets/dataset_IXI_T1_brain_extraction_registered_resampled_2nd_pass/"

filelist_ixi = os.listdir(BETTIK_DIR + ixi_dir)
filelist_ixi = [ixi_dir + item for item in filelist_ixi]

random.shuffle(filelist_ixi)

cutoff = int(0.8 * len(filelist_ixi))

train_sublist = filelist_ixi[:cutoff]

val_sublist = filelist_ixi[cutoff:cutoff+(len(filelist_ixi)-cutoff)//2]

test_sublist = filelist_ixi[cutoff+(len(filelist_ixi)-cutoff)//2:]



with open("train.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in train_sublist:
        writer.writerow([item])

with open("val.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in val_sublist:
        writer.writerow([item])

with open("test.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in test_sublist:
        writer.writerow([item])
