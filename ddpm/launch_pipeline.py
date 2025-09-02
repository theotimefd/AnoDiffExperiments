
import os
from train_ddpm import launch_train
from compute_metrics_reconstruction import launch_compute_metrics_reconstruction

def main():
    parser.add_argument(
        "-c",
        "--config-file",
        default="/config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )

    parser = argparse.ArgumentParser(description="DDPM training script")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    
    args = parser.parse_args()
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in config_dict.items():
        setattr(args, k, v)

    os.makedirs(f"{args.root_dir}/AnoDiffExperiments/{config_dict['experiment_name']}/{config_dict['sub_experiment_name']}/models/", exist_ok=True)
    os.makedirs(f"{args.root_dir}/AnoDiffExperiments/tensorboard/{config_dict['sub_experiment_name']}/", exist_ok=True)

    print(f"Launching training: {config_dict['experiment_name']}/{config_dict['sub_experiment_name']} with {args.gpus} gpus")
    
    if args.diffusion_train["enabled"]:
        launch_train(args)
    if args.compute_metrics_reconstruction["enabled"]:
        launch_compute_metrics_reconstruction(args)


if __name__ == "__main__":
    main()