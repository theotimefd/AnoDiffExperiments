"""Generates a flowchart of the 2D DDPM anomaly detection framework and saves it as PNG."""
import graphviz

dot = graphviz.Digraph(
    "2D_DDPM_Framework",
    format="png",
    graph_attr={
        "rankdir": "TB",
        "bgcolor": "white",
        "fontname": "Helvetica",
        "fontsize": "14",
        "pad": "0.5",
        "nodesep": "0.6",
        "ranksep": "0.8",
        "dpi": "150",
        "label": "2D DDPM Anomaly Detection Framework\n(launch_pipeline.py)",
        "labelloc": "t",
        "labeljust": "c",
        "fontsize": "20",
        "fontcolor": "#1a1a2e",
    },
    node_attr={
        "fontname": "Helvetica",
        "fontsize": "11",
        "style": "filled",
        "shape": "box",
        "penwidth": "1.5",
    },
    edge_attr={
        "fontname": "Helvetica",
        "fontsize": "10",
        "color": "#555555",
        "arrowsize": "0.8",
    },
)

# ── Cluster: Configuration ──────────────────────────────────────────────
with dot.subgraph(name="cluster_config") as c:
    c.attr(label="Configuration", style="rounded,dashed", color="#888888", fontcolor="#555555")
    c.node("config", "config.json\n(experiment, dataset,\nnoise, network, training params)", fillcolor="#e8f4f8", color="#4a90d9", shape="note")

# ── Entry point ─────────────────────────────────────────────────────────
dot.node("launch", "launch_pipeline.py\n(main entry point)", fillcolor="#ffeaa7", color="#d4a017", shape="box", penwidth="2.5")

# ── Cluster: Training ──────────────────────────────────────────────────
with dot.subgraph(name="cluster_train") as c:
    c.attr(label="1 — Training Phase", style="rounded,filled", color="#27ae60", fillcolor="#eafaf1", fontcolor="#27ae60", fontsize="13")
    c.node("train", "train_ddpm.py\n(2D slice training)", fillcolor="#a3e4d7", color="#1e8449")
    c.node("train_fv", "train_ddpm_full_volume.py\n(2D slices, full-volume val)", fillcolor="#a3e4d7", color="#1e8449")

    # Sub-details of training
    c.node("data_load", "Load Train/Val Data\n(CacheDataset + DataLoader)", fillcolor="#d5f5e3", color="#1e8449", shape="box")
    c.node("transforms", "Apply Transforms\n(ScaleIntensity, RandAffine,\nGet2DSlice, Crop, Flip…)", fillcolor="#d5f5e3", color="#1e8449", shape="box")
    c.node("unet", "DiffusionModelUNet\n(MONAI)", fillcolor="#d5f5e3", color="#1e8449", shape="component")
    c.node("noise_gen", "Noise Generation\n(Simplex or Gaussian)", fillcolor="#d5f5e3", color="#1e8449", shape="box")
    c.node("loss", "Compute Loss\n(MSE: predicted vs actual noise)", fillcolor="#d5f5e3", color="#1e8449", shape="box")
    c.node("best_model", "Save Best Model\n(.pth checkpoint)", fillcolor="#a3e4d7", color="#1e8449", shape="folder")

# ── Cluster: Reconstruction Metrics ────────────────────────────────────
with dot.subgraph(name="cluster_recon") as c:
    c.attr(label="2 — Reconstruction Metrics", style="rounded,filled", color="#2980b9", fillcolor="#ebf5fb", fontcolor="#2980b9", fontsize="13")
    c.node("recon", "compute_metrics_reconstruction.py", fillcolor="#aed6f1", color="#2471a3")
    c.node("sample_recon", "sample.py → my_sample()\nDenoise healthy test images", fillcolor="#d6eaf8", color="#2471a3", shape="box")
    c.node("recon_metrics", "Metrics: PSNR, SSIM,\nMSE, LPIPS", fillcolor="#d6eaf8", color="#2471a3", shape="box")

# ── Cluster: Anomaly Detection ─────────────────────────────────────────
with dot.subgraph(name="cluster_ano") as c:
    c.attr(label="3 — Anomaly Detection Inference", style="rounded,filled", color="#c0392b", fillcolor="#fdedec", fontcolor="#c0392b", fontsize="13")
    c.node("ano_infer", "anomaly_detection_inference.py", fillcolor="#f5b7b1", color="#c0392b")
    c.node("sample_ano", "sample.py → my_sample()\nor sample_thor()\n(Denoise pathological images\nslice by slice)", fillcolor="#fadbd8", color="#c0392b", shape="box")
    c.node("anomaly_map", "make_anomaly_maps.py\n|original − reconstructed|", fillcolor="#fadbd8", color="#c0392b", shape="box")

# ── Cluster: Anomaly Metrics ───────────────────────────────────────────
with dot.subgraph(name="cluster_ano_metrics") as c:
    c.attr(label="4 — Anomaly Detection Metrics", style="rounded,filled", color="#8e44ad", fillcolor="#f4ecf7", fontcolor="#8e44ad", fontsize="13")
    c.node("ano_metrics", "compute_metrics_anomaly_detection.py", fillcolor="#d2b4de", color="#7d3c98")
    c.node("postproc", "Post-processing\n(threshold, median filter,\nerosion/dilation, fill holes)", fillcolor="#e8daef", color="#7d3c98", shape="box")
    c.node("scores", "Scores: IoU, Dice,\nHausdorff, Precision,\nRecall, F1", fillcolor="#e8daef", color="#7d3c98", shape="box")

# ── Cluster: Parameter Selection ───────────────────────────────────────
with dot.subgraph(name="cluster_params") as c:
    c.attr(label="5 — Parameter Selection (CPU)", style="rounded,filled", color="#e67e22", fillcolor="#fef5e7", fontcolor="#e67e22", fontsize="13")
    c.node("select_params", "compute_select_params_cpu.py\n(select best noise timestep\n& threshold on 50% test data)", fillcolor="#fad7a0", color="#ca6f1e", shape="box")

# ── EDGES ───────────────────────────────────────────────────────────────

# Config → launcher
dot.edge("config", "launch", label="  parse args", style="dashed")

# launcher → pipeline steps
dot.edge("launch", "train", label="step: train_ddpm")
dot.edge("launch", "train_fv", label="step: train_ddpm_full_volume")
dot.edge("launch", "recon", label="step: compute_metrics_reconstruction")
dot.edge("launch", "ano_metrics", label="step: compute_metrics\n_anomaly_detection")
dot.edge("launch", "select_params", label="step: compute_select\n_params_cpu")
dot.edge("launch", "ano_infer", label="step: anomaly_detection\n_inference")

# Training internals
dot.edge("train", "data_load")
dot.edge("train_fv", "data_load")
dot.edge("data_load", "transforms")
dot.edge("transforms", "noise_gen")
dot.edge("noise_gen", "unet", label="noised image")
dot.edge("unet", "loss", label="predicted noise")
dot.edge("noise_gen", "loss", label="actual noise", style="dashed")
dot.edge("loss", "best_model", label="save on\nbest val loss")

# Reconstruction metrics
dot.edge("best_model", "recon", label="load model", style="dotted")
dot.edge("recon", "sample_recon")
dot.edge("sample_recon", "recon_metrics")

# Anomaly detection inference
dot.edge("best_model", "ano_infer", label="load model", style="dotted")
dot.edge("ano_infer", "sample_ano")
dot.edge("sample_ano", "anomaly_map")

# Anomaly metrics
dot.edge("best_model", "ano_metrics", label="load model", style="dotted")
dot.edge("ano_metrics", "sample_ano", style="dashed", label="  uses")
dot.edge("anomaly_map", "postproc")
dot.edge("ano_metrics", "postproc", style="dashed")
dot.edge("postproc", "scores")

# Parameter selection
dot.edge("select_params", "scores", style="dashed", label="  uses scores to\n  pick best params")

# Output
output_path = "/home/rivage/bettik/AnoDiffExperiments/2d_ddpm/2d_ddpm_framework_flowchart"
dot.render(output_path, cleanup=True)
print(f"Flowchart saved to {output_path}.png")
