import os
import csv
import random
# Set the random seed for reproducibility
random.seed(42)

#BETTIK_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
#BETTIK_DIR = "/home/theotime/bettik/"
BETTIK_DIR = "/home/fehrdelt/bettik/"

aini_stroke_adc_unhealthy_dir = "datasets/Aini-Stroke_ADC/Others_extracted_brain/"

filelist_aini_stroke_adc_unhealthy = os.listdir(BETTIK_DIR+aini_stroke_adc_unhealthy_dir)
filelist_aini_stroke_adc_unhealthy = [aini_stroke_adc_unhealthy_dir+item for item in filelist_aini_stroke_adc_unhealthy]

with open("ADC/aini-stroke_unhealthy_test.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for item in filelist_aini_stroke_adc_unhealthy:
        writer.writerow([item])