
import os
import argparse
import json
from pathlib import Path
from train_autoencoder import launch_train_autoencoder
from train_diffusion import launch_train_diffusion
from compute_metrics_reconstruction_ae import launch_compute_metrics_reconstruction_ae
from compute_metrics_anomaly_detection_ae import launch_compute_metrics_anomaly_detection_ae


def main():
    print("start of main")
    parser = argparse.ArgumentParser(description="3D LDM training script")
    parser.add_argument(
        "-c",
        "--config-file",
        default="/config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )

    
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    
    args = parser.parse_args()
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in config_dict.items():
        setattr(args, k, v)
    
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel

    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1
        device = 0

    if rank == 0:
        
        print("making directories")
        os.makedirs(f"{args.root_dir}/AnoDiffExperiments/{config_dict['experiment_name']}/{config_dict['sub_experiment_name']}/models/", exist_ok=True)
        os.makedirs(f"{args.root_dir}/AnoDiffExperiments/tensorboard/{config_dict['sub_experiment_name']}/", exist_ok=True)


    for step in args.pipeline:
        print(step)
        
        if step == "train_autoencoder":
            print(f"Launching autoencoder training: {config_dict['experiment_name']}/{config_dict['sub_experiment_name']} with {args.gpus} gpus")
            launch_train_autoencoder(args)
        
        if step == "train_diffusion":
            print(f"Launching diffusion training: {config_dict['experiment_name']}/{config_dict['sub_experiment_name']} with {args.gpus} gpus")
            launch_train_diffusion(args)

        if step == "compute_metrics_reconstruction_ae" and rank == 0:
            print("Launching reconstruction metrics computation")
            launch_compute_metrics_reconstruction_ae(args)

        if step == "compute_metrics_anomaly_detection_ae" and rank == 0:
            print("Launching anomaly detection metrics computation for autoencoder")
            launch_compute_metrics_anomaly_detection_ae(args)
        
            
    


if __name__ == "__main__":
    main()