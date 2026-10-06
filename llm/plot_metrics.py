import argparse
import json
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def smooth_curve(values: List[float], weight: float = 0.8) -> List[float]:
    """Exponential moving average for smoother loss curves."""
    if not values:
        return []
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_metrics(
    metrics_data: Dict,
    save_path: str = "training_curves.png",
    smooth_weight: float = 0.6,
):
    """
    Generates a clean multi-panel plot of the training dynamics:
    - Stage 1: L2 Prediction Loss and SIGReg Regularizer
    - Stage 2: Cross-Entropy Loss and Validation Perplexity
    """
    has_stage1 = "stage1" in metrics_data and len(metrics_data["stage1"].get("train_steps", [])) > 0
    has_stage2 = "stage2" in metrics_data and len(metrics_data["stage2"].get("train_steps", [])) > 0
    has_baseline = "baseline" in metrics_data and len(metrics_data["baseline"].get("train_steps", [])) > 0

    if not (has_stage1 or has_stage2 or has_baseline):
        print(f"[Plotter] No valid metric steps found to plot in {save_path}")
        return

    # Determine layout
    if has_stage1 and has_stage2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Panel 1: Stage 1 L2 Pred Loss
        s1 = metrics_data["stage1"]
        steps = s1["train_steps"]
        pred_loss = s1["pred_loss"]
        axes[0].plot(steps, pred_loss, alpha=0.3, color="royalblue", label="Raw L2 Pred Loss")
        axes[0].plot(steps, smooth_curve(pred_loss, smooth_weight), color="royalblue", lw=2, label="Smoothed")
        if "val_steps" in s1 and s1["val_steps"]:
            axes[0].plot(s1["val_steps"], s1["val_pred_loss"], "o-", color="darkblue", label="Val L2 Pred")
        axes[0].set_title("Stage 1: LeJEPA L2 Latent Prediction Loss", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("MSE Loss")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()

        # Panel 2: Stage 1 SIGReg Loss
        sigreg_loss = s1["sigreg_loss"]
        axes[1].plot(steps, sigreg_loss, alpha=0.3, color="darkorange", label="Raw SIGReg")
        axes[1].plot(steps, smooth_curve(sigreg_loss, smooth_weight), color="darkorange", lw=2, label="Smoothed")
        if "val_steps" in s1 and s1["val_steps"]:
            axes[1].plot(s1["val_steps"], s1["val_sigreg_loss"], "o-", color="firebrick", label="Val SIGReg")
        axes[1].set_title("Stage 1: SIGReg Regularizer (Collapse Prevention)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Epps-Pulley Statistic")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

        # Panel 3: Stage 2 Cross-Entropy Loss
        s2 = metrics_data["stage2"]
        steps2 = s2["train_steps"]
        ce_loss = s2["ce_loss"]
        axes[2].plot(steps2, ce_loss, alpha=0.3, color="forestgreen", label="Train CE Loss")
        axes[2].plot(steps2, smooth_curve(ce_loss, smooth_weight), color="forestgreen", lw=2, label="Smoothed Train")
        if "val_steps" in s2 and s2["val_steps"]:
            axes[2].plot(s2["val_steps"], s2["val_ce_loss"], "o-", color="crimson", lw=2, label="Validation CE Loss")
        axes[2].set_title("Stage 2: Conditional Decoder Cross-Entropy Loss", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Cross-Entropy Loss")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend()

        # Panel 4: Stage 2 Validation Perplexity
        if "val_steps" in s2 and s2["val_steps"]:
            axes[3].plot(s2["val_steps"], s2["val_ppl"], "s-", color="purple", lw=2, label="Val Perplexity")
            axes[3].set_title("Stage 2: Validation Perplexity (PPL)", fontsize=12, fontweight="bold")
            axes[3].set_xlabel("Step")
            axes[3].set_ylabel("Perplexity")
            axes[3].grid(True, linestyle="--", alpha=0.5)
            axes[3].legend()
        else:
            train_ppl = s2.get("train_ppl", [])
            if train_ppl:
                axes[3].plot(steps2, smooth_curve(train_ppl, smooth_weight), color="purple", lw=2, label="Train PPL")
                axes[3].set_title("Stage 2: Training Perplexity", fontsize=12, fontweight="bold")
                axes[3].set_xlabel("Step")
                axes[3].set_ylabel("Perplexity")
                axes[3].grid(True, linestyle="--", alpha=0.5)
                axes[3].legend()

        plt.suptitle("Hierarchical LeJEPA Language Model Training Dynamics", fontsize=14, fontweight="bold", y=0.98)
    elif has_baseline:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        b = metrics_data["baseline"]
        steps = b["train_steps"]
        ce_loss = b["ce_loss"]
        axes[0].plot(steps, ce_loss, alpha=0.3, color="teal", label="Train CE Loss")
        axes[0].plot(steps, smooth_curve(ce_loss, smooth_weight), color="teal", lw=2, label="Smoothed Train")
        if "val_steps" in b and b["val_steps"]:
            axes[0].plot(b["val_steps"], b["val_ce_loss"], "o-", color="red", label="Val CE Loss")
        axes[0].set_title("Baseline Transformer: Cross-Entropy Loss", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()

        if "val_steps" in b and b["val_steps"]:
            axes[1].plot(b["val_steps"], b["val_ppl"], "s-", color="purple", lw=2, label="Val Perplexity")
            axes[1].set_title("Baseline Transformer: Validation Perplexity", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("PPL")
            axes[1].grid(True, linestyle="--", alpha=0.5)
            axes[1].legend()
        plt.suptitle("Standard Transformer Language Model Training", fontsize=14, fontweight="bold")
    else:
        # Only stage 1 or stage 2
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # fallback simple plot
        pass

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plotter] Saved plot to: {save_path}")


def plot_comparison_metrics(
    compare_data: Dict,
    save_path: str = "comparison_curves.png",
    smooth_weight: float = 0.6,
):
    """
    Plots side-by-side comparison curves between LeJEPA LM and Standard Transformer LM.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    jepa_s2 = compare_data.get("lejepa_stage2", {})
    base = compare_data.get("baseline", {})

    # Left: Training CE Loss
    if "train_steps" in jepa_s2 and jepa_s2["train_steps"]:
        axes[0].plot(
            jepa_s2["train_steps"],
            smooth_curve(jepa_s2["ce_loss"], smooth_weight),
            color="forestgreen",
            lw=2.5,
            label="LeJEPA Conditional Decoder",
        )
    if "train_steps" in base and base["train_steps"]:
        axes[0].plot(
            base["train_steps"],
            smooth_curve(base["ce_loss"], smooth_weight),
            color="royalblue",
            lw=2.5,
            label="Standard Transformer Baseline",
        )

    axes[0].set_title("Training Cross-Entropy Loss: LeJEPA vs Baseline", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Decoder Step")
    axes[0].set_ylabel("CE Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Right: Validation Perplexity
    val_labels = []
    val_ppls = []
    colors = []

    if "val_ppl" in compare_data.get("summary", {}).get("lejepa", {}):
        val_labels.append("LeJEPA LM\n(Hierarchical)")
        val_ppls.append(compare_data["summary"]["lejepa"]["val_ppl"])
        colors.append("forestgreen")

    if "val_ppl" in compare_data.get("summary", {}).get("baseline", {}):
        val_labels.append("Standard LM\n(Baseline)")
        val_ppls.append(compare_data["summary"]["baseline"]["val_ppl"])
        colors.append("royalblue")

    if val_labels:
        bars = axes[1].bar(val_labels, val_ppls, color=colors, width=0.45)
        axes[1].set_title("Validation Perplexity Comparison (Lower is Better)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Perplexity")
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plotter] Saved comparison plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from saved JSON log file")
    parser.add_argument("--log_file", type=str, required=True, help="Path to JSON metrics log file")
    parser.add_argument("--save_path", type=str, default=None, help="Output image file path (e.g. plot.png)")
    parser.add_argument("--smooth", type=float, default=0.6, help="Smoothing weight (0.0 = none, 0.9 = heavy)")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        raise FileNotFoundError(f"Log file not found: {args.log_file}")

    with open(args.log_file, "r") as f:
        data = json.load(f)

    out_path = args.save_path
    if out_path is None:
        base, _ = os.path.splitext(args.log_file)
        out_path = base + "_plot.png"

    if "lejepa_stage2" in data and "baseline" in data:
        plot_comparison_metrics(data, save_path=out_path, smooth_weight=args.smooth)
    else:
        plot_training_metrics(data, save_path=out_path, smooth_weight=args.smooth)


if __name__ == "__main__":
    main()
