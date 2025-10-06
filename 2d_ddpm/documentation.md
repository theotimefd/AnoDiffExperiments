

### compute_metrics_reconstruction
if the validation set is in 2D: it will noise the 2D image, denoise it with the ddpm and compute image similarity metrics (psnr, ssim, mse & lpips)
if the validation set is in 3D, it will noise the 3D image, and denoise it slice by slice using the 2D ddpm and compute image similarity metrics

### compute_metrics_anomaly_detection
pass

### config.json
{
    "experiment_name": "experiment_1",
    "sub_experiment_name": "exp_1_0",
    "description": "2D DDPM on flair with simplex noise, validation on whole volume",
    "pipeline": ["compute_metrics_reconstruction"]
        **list of steps to exectute. "train_ddpm" : 2d ddpm with single slice validation**
                                     **"train_ddpm_full_volume": 2d ddpm with whole volume validation slice by slice**
                                     **"compute_metrics_reconstruction"**
                                     **"compute_metrics_anomaly_detection"**
                                     **"compute_metrics_thor_anomaly_detection**
    "spatial_dims": 2,
    "spatial_dims_val_test": 3 **2 for single slice validation & test (metrics recons & ano detec), 3 for full volume slice by slice validation & test (metrics recons & ano detec)**
    "image_size": 128,
    "root_dir":"/bettik/PROJECTS/pr-gin5_aini/fehrdelt/",
    "dataset":{
        "name": "final_flair_dataset_small_added_oasis", **name of the dataset for train, val and test_reconstruction**
        "test": "brats", **name of the dataset for anomaly detection**
        "batch_size": 16,
        "num_workers": 16,
        "slice_indexes_start":22,
        "slice_indexes_end": 94
    },
    "slice_indexes_start":22,
    "slice_indexes_end": 94,
    "train_transforms": {
        "_target_": "monai.transforms.Compose",
        "transforms": [
        {
            "_target_": "monai.transforms.LoadImage", "image_only": true
        },
        {
            "_target_": "monai.transforms.EnsureChannelFirst"
        },
        {
            "_target_": "monai.transforms.RandAffine", "prob": 0.5, "rotate_range": [0.1, 0.1, 0.1]
        },
        { **makes sure the peak of intensities from the histogram is always at the same value**
          **! do it on the whole volume before selecting a slice !**
            "_target_": "utils.custom_transforms.ScaleIntensityFromHistogramPeak", "target_value": 200.0
        },
        {
            "_target_": "utils.custom_transforms.Get2DSliceFromIndexes", "axis": 2, "indexes_start": "@slice_indexes_start", "indexes_end": "@slice_indexes_end"
        },
        {
            "_target_": "monai.transforms.RandScaleCrop", "roi_scale":0.9, "max_roi_scale":1.1, "random_size":true
        },
        {
            "_target_": "monai.transforms.ResizeWithPadOrCrop", "spatial_size":["@image_size", "@image_size"]
        },
        {
            "_target_": "monai.transforms.ScaleIntensityRange", "a_min":0.0, "a_max":700.0, "b_min":0.0, "b_max":1.0, "clip":true
        },
        {
            "_target_": "monai.transforms.RandFlip", "prob": 0.5, "spatial_axis": 0
        },
        {
            "_target_": "utils.custom_transforms.SetBackgroundToZero"
        }]
    },
    "val_transforms": {
        "_target_": "monai.transforms.Compose",
        "transforms": [
            {
                "_target_": "monai.transforms.LoadImage", "image_only": true
            },
            {
                "_target_": "monai.transforms.EnsureChannelFirst"
            },
            {
                "_target_": "monai.transforms.ResizeWithPadOrCrop", "spatial_size":["@image_size", "@image_size", "@image_size"]
            },
            {
                "_target_": "utils.custom_transforms.ScaleIntensityFromHistogramPeak", "target_value": 200.0
            },
            {
                "_target_": "monai.transforms.ScaleIntensityRange", "a_min":0.0, "a_max":700.0, "b_min":0.0, "b_max":1.0, "clip":true
            },
            {
                "_target_": "utils.custom_transforms.SetBackgroundToZero"
            }
        ]
    },
    "noise": {
        "type": "simplex",
        "schedule":"cosine",
        "simplex_octaves":6, 
        "simplex_persistence":0.8, 
        "simplex_frequency":64,
        "normalize": false,
        "num_timesteps_full_noise": 1000,
        "noise_rate_train_and_infer": 0.35
    },
    "network_def": {
        "_target_": "monai.networks.nets.DiffusionModelUNet",
        "spatial_dims": "@spatial_dims",
        "in_channels": 1,
        "out_channels": 1,
        "channels":[32, 64, 64, 64],
        "attention_levels":[false, true, true, true],
        "num_head_channels":8,
        "use_flash_attention":true
    },
    "diffusion_train": {
        "max_epochs": 6000,
        "val_interval": 10,
        "optimizer": {
            "type": "Adam",
            "lr": 1.4e-5
        },
        "lr_scheduler": "none",
        "lr_scheduler_milestones": [1000],
        "ema":true
    },
    "compute_metrics_reconstruction":{
        "transforms": {
            "transform_0":{
                "_target_": "transforms.LoadImage", "image_only": true
            },
            "transform_1":{
                "_target_": "transforms.EnsureChannelFirst"
            },
            "transform_3":{
                "_target_": "custom_transforms.Get2DSlice", "axis": 2
            },
            "transform_5":{
                "_target_": "transforms.ResizeWithPadOrCrop", "spatial_size":["@image_size", "@image_size"]
            },
            "transform_6":{
                "_target_": "custom_transforms.ScaleIntensityFromHistogramPeak", "target_value": 200.0
            },
            "transform_7":{
                "_target_": "transforms.ScaleIntensityRange", "a_min":0.0, "a_max":700.0, "b_min":0.0, "b_max":1.0, "clip":true
            },
            "transform_9":{
                "_target_": "custom_transforms.SetBackgroundToZero"
            }
        },
        "noise_rate_min": 0.15,
        "noise_rate_max": 0.35,
        "noise_timesteps_interval": 50,
        "noise_rate_visualize":0.35
    }
}
