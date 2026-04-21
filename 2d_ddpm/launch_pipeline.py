import sys
sys.path.append("../..")
import os
import argparse
import json
from pathlib import Path

from utils.utils import dtprint

def main():

    dtprint(f"launch_pipeline.py main func")

    parser = argparse.ArgumentParser(description="2D DDPM training script")
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

        os.makedirs(f"{args.root_dir}/AnoDiffExperiments/{config_dict['experiment_name']}/{config_dict['sub_experiment_name']}/models/", exist_ok=True)
        os.makedirs(f"{args.root_dir}/AnoDiffExperiments/tensorboard/{config_dict['sub_experiment_name']}/", exist_ok=True)

    

    for step in args.pipeline:
        
        if step == "train_ddpm":
            from train_ddpm import launch_train
            dtprint(f"Launching ddpm training: {config_dict['experiment_name']}/{config_dict['sub_experiment_name']} with {args.gpus} gpus")
            launch_train(args)

        if step == "train_ddpm_full_volume":
            from train_ddpm_full_volume import launch_train_full_volume
            dtprint(f"Launching ddpm full volume training: {config_dict['experiment_name']}/{config_dict['sub_experiment_name']} with {args.gpus} gpus")
            launch_train_full_volume(args)
        
        if step == "compute_metrics_reconstruction" and rank == 0:
            from compute_metrics_reconstruction import launch_compute_metrics_reconstruction
            dtprint("Launching compute_metrics_reconstruction")
            launch_compute_metrics_reconstruction(args)
        
        if step=="compute_metrics_anomaly_detection" and rank==0:
            from compute_metrics_anomaly_detection import launch_compute_metrics_anomaly_detection
            dtprint("Launching compute_metrics_anomaly_detection")
            launch_compute_metrics_anomaly_detection(args)
        
        if step=="compute_select_params_cpu" and rank==0:
            from utils.compute_select_params_cpu import launch_compute_select_params_cpu
            dtprint("Launching compute_select_params_cpu")
            launch_compute_select_params_cpu(args)
        
        if step=="anomaly_detection_inference" and rank==0:
            from anomaly_detection_inference import launch_anomaly_detection_inference
            dtprint("Launching anomaly_detection_inference")
            launch_anomaly_detection_inference(args)
        
        if step=="anomaly_detection_inference_no_abs_value" and rank==0:
            from anomaly_detection_inference import launch_anomaly_detection_inference
            dtprint("Launching anomaly_detection_inference with no abs value for the anomaly maps")
            launch_anomaly_detection_inference(args, no_abs_value=True)
        
        if step=="anomaly_detection_inference_20x" and rank==0:
            from anomaly_detection_inference import launch_anomaly_detection_inference
            dtprint("Launching anomaly_detection_inference with 20x inferences")
            launch_anomaly_detection_inference(args, nb_inferences=20)
        
        if step=="anomaly_detection_inference_20x_no_abs_value" and rank==0:
            from anomaly_detection_inference import launch_anomaly_detection_inference
            dtprint("Launching anomaly_detection_inference with 20x inferences and no abs value for the anomaly maps")
            launch_anomaly_detection_inference(args, no_abs_value=True, nb_inferences=20)
        


if __name__ == "__main__":
    main()