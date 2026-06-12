import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import pearsonr
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
import multiprocessing
from functools import partial

def compute_ncc(image_data, template_data):
    """Compute Normalized Cross-Correlation (NCC) / Pearson correlation."""
    img_flat = image_data.flatten()
    temp_flat = template_data.flatten()
    
    # Ignore absolute zeros which might be background for both
    mask = (img_flat != 0) | (temp_flat != 0)
    
    if np.sum(mask) < 2:
        return 0.0
        
    corr, _ = pearsonr(img_flat[mask], temp_flat[mask])
    return corr

def compute_nmi(image_data, template_data, bins=100):
    """Compute Normalized Mutual Information (NMI)."""
    img_flat = image_data.flatten()
    temp_flat = template_data.flatten()
    
    mask = (img_flat != 0) | (temp_flat != 0)
    
    if np.sum(mask) < 2:
        return 0.0
        
    img_masked = img_flat[mask]
    temp_masked = temp_flat[mask]
    
    # Digitize continuous intensities into discrete bins
    img_binned = np.digitize(img_masked, bins=np.histogram_bin_edges(img_masked, bins=bins))
    temp_binned = np.digitize(temp_masked, bins=np.histogram_bin_edges(temp_masked, bins=bins))
    
    nmi = normalized_mutual_info_score(temp_binned, img_binned)
    return nmi

def _process_single_file(file_path, template_data):
    """Worker function to process a single NIfTI file."""
    try:
        img = nib.load(file_path)
        img_data = img.get_fdata()

        hist_norm_target_value = 200
        # Compute the histogram of the image slice
        hist, bins = np.histogram(img_data.flatten(), bins=100, range=(np.max(img_data)/5.0, np.max(img_data)))
        # Find the value corresponding to the maximum of the histogram
        most_occurred_pixel_value = bins[np.argmax(hist)]
        norm_img_data = img_data/most_occurred_pixel_value*hist_norm_target_value
        
        if norm_img_data.shape != template_data.shape:
            return {
                'filename': os.path.basename(file_path),
                'ncc_score': np.nan,
                'nmi_score': np.nan,
                'error': f"Shape mismatch: {norm_img_data.shape} != {template_data.shape}"
            }
            
        ncc_score = compute_ncc(norm_img_data, template_data)
        nmi_score = compute_nmi(norm_img_data, template_data)
        
        return {
            'filename': os.path.basename(file_path),
            'ncc_score': ncc_score,
            'nmi_score': nmi_score,
            'error': None
        }
    except Exception as e:
        return {
            'filename': os.path.basename(file_path),
            'ncc_score': np.nan,
            'nmi_score': np.nan,
            'error': str(e)
        }

def compute_registration_qc(folder_path, template_path, output_csv=None, num_workers=None):
    """
    Computes registration quality control metrics (NCC and NMI) between NIfTI files in a folder and a template.
    Uses multiprocessing for faster execution.
    
    Args:
        folder_path (str): Path to the folder containing NIfTI files.
        template_path (str): Path to the MNI template NIfTI file.
        output_csv (str, optional): Path to save the resulting dataframe as CSV.
        num_workers (int, optional): Number of CPU cores to use. Defaults to all available cores.
        
    Returns:
        pd.DataFrame: DataFrame containing filenames and overlap scores.
    """
    print("using hist norm")
    print(f"Loading template: {template_path}")
    template_img = nib.load(template_path)
    template_data = template_img.get_fdata()
    
    file_pattern = os.path.join(folder_path, '*.nii*')
    nifti_files = glob.glob(file_pattern)
    
    if not nifti_files:
        print(f"No NIfTI files found in {folder_path}")
        return pd.DataFrame()
        
    if num_workers is None:
        num_workers = min(32, multiprocessing.cpu_count())
        
    print(f"Found {len(nifti_files)} NIfTI files. Computing metrics using {num_workers} processes...")
    results = []
    
    worker = partial(_process_single_file, template_data=template_data)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(worker, nifti_files), total=len(nifti_files)):
            if result['error'] is not None:
                print(f"Error processing {result['filename']}: {result['error']}")
            results.append({
                'filename': result['filename'],
                'ncc_score': result['ncc_score'],
                'nmi_score': result['nmi_score']
            })
            
    df = pd.DataFrame(results)
    
    if output_csv is not None and not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")
        
    return df


def get_nifti_list_by_registration_quality(folder_path, template_path, num_workers=None):

    qc_df = compute_registration_qc(folder_path, template_path, num_workers = num_workers)
    
    qc_df_sorted = qc_df.sort_values(by=['ncc_score'], ignore_index=True, ascending=True)

    nifti_list = qc_df_sorted['filename'].tolist()
    
    return nifti_list


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute registration QC metrics against an MNI template using multiprocessing")
    parser.add_argument("--input_folder", required=True, help="Folder containing registered NIfTI files")
    parser.add_argument("--template", required=True, help="Path to the MNI template NIfTI file")
    parser.add_argument("--output", help="Optional CSV file to save results")
    parser.add_argument("--workers", type=int, default=None, help="Number of concurrent workers (default: use all CPU cores)")
    
    args = parser.parse_args()
    
    df = compute_registration_qc(args.input_folder, args.template, args.output, num_workers=args.workers)
    
    if not df.empty:
        print("\nResults:")
        print(df.to_string())
