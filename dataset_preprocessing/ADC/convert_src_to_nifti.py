import nibabel as nib
import scipy.io
import subprocess
import os

DATA_DIR = '/data_network/summer_projects/fehrdelt/Current/2024_these_FEHR--DELUDE_Theotime/Databases/Human_Connectome_Project_Young_Adult_HCP-YA'
Unzipped_DATA_DIR = '/data_network/summer_projects/fehrdelt/Current/2024_these_FEHR--DELUDE_Theotime/Databases/mat_Human_Connectome_Project_Young_Adult_HCP-YA'



for file in os.listdir(DATA_DIR):
    #print(f"file {i+1}/{len(files)}: {file}")
    if file.endswith('.sz'):
        #subprocess.run(["gunzip", "-c", os.path.join(DATA_DIR, file), '>', os.path.join(Unzipped_DATA_DIR, file)])
        p = subprocess.Popen([f'gunzip -c {os.path.join(DATA_DIR, file)} > {os.path.join(Unzipped_DATA_DIR, f"{file[:-2]}mat")}'], shell=True) 
        p.wait()
     