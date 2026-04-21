
import os
import argparse
import json
from pathlib import Path
from train_3d_ddpm_patch import launch_train_patch
from compute_metrics_reconstruction import launch_compute_metrics_reconstruction
from compute_metrics_anomaly_detection import launch_compute_metrics_anomaly_detection
from utils.compute_select_params_cpu import launch_compute_select_params_cpu
from anomaly_detection_inference import launch_anomaly_detection_inference
from anomaly_detection_inference_20x import launch_anomaly_detection_inference as launch_anomaly_detection_inference_20x
#from compute_metrics_thor_anomaly_detection import launch_compute_metrics_thor_anomaly_detection
from utils.utils import dtprint

def main():
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
            dtprint(f"Launching ddpm training: {config_dict['experiment_name']}/{config_dict['sub_experiment_name']} with {args.gpus} gpus")
            launch_train_patch(args)
        
        if step == "compute_metrics_reconstruction" and rank == 0:
            dtprint("Launching compute_metrics_reconstruction")
            launch_compute_metrics_reconstruction(args)
        
        
        if step=="compute_metrics_anomaly_detection" and rank==0:
            dtprint("Launching compute_metrics_anomaly_detection")
            launch_compute_metrics_anomaly_detection(args)
        
        if step=="compute_select_params_cpu" and rank==0:
            dtprint("Launching compute_select_params_cpu")
            launch_compute_select_params_cpu(args)
        
        """
        if step=="compute_metrics_thor_anomaly_detection" and rank==0:
            dtprint("Launching compute_metrics_thor_anomaly_detection")
            launch_compute_metrics_thor_anomaly_detection(args)
        """
        if step=="anomaly_detection_inference" and rank==0:
            dtprint("Launching anomaly_detection_inference")
            launch_anomaly_detection_inference(args)
        
        if step=="anomaly_detection_inference_no_abs_value" and rank==0:
            dtprint("Launching anomaly_detection_inference with no abs value for the anomaly maps")
            launch_anomaly_detection_inference(args, no_abs_value=True)
                
        if step=="anomaly_detection_inference_20x" and rank==0:
            dtprint("Launching anomaly_detection_inference with 20x inferences")
            launch_anomaly_detection_inference_20x(args)
        
        if step=="anomaly_detection_inference_20x_no_abs_value" and rank==0:
            dtprint("Launching anomaly_detection_inference with 20x inferences and no abs value for the anomaly maps")
            launch_anomaly_detection_inference_20x(args, no_abs_value=True)
        
        


if __name__ == "__main__":
    main()