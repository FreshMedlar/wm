import argparse
import json
import math
import os
import time
from typing import Optional, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm.dataset import get_dataloaders
from llm.models import LeJEPALanguageModel, StandardTransformerLM
from llm.plot_metrics import plot_training_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Stage LeJEPA Language Model P(x_{t+1} | x_{\le t}, z_hat_{t+1}) & Baseline Transformer"
    )

    # Model Architecture
    parser.add_argument(
        "--model",
        type=str,
        default="lejepa",
        choices=["lejepa", "baseline"],
        help="Model architecture: 'lejepa' (hierarchical planner + decoder) or 'baseline' (standard LLM)",
    )
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size (GPT-2 default)")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--d_latent", type=int, default=128, help="Continuous latent space dimension for LeJEPA")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--lejepa_encoder_layers", type=int, default=7, help="Number of layers in LeJEPA causal encoder")
    parser.add_argument("--lejepa_predictor_layers", type=int, default=2, help="Number of layers in LeJEPA latent predictor")
    parser.add_argument("--decoder_layers", type=int, default=1, help="Number of layers in Conditional Decoder (~10% budget)")
    parser.add_argument("--baseline_layers", type=int, default=10, help="Number of layers in standard baseline transformer")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--sigreg_weight", type=float, default=0.1, help="Weight lambda for SIGReg isotropic regularizer")

    # Training Scheme & Dynamics
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "1", "2", "baseline"],
        help="Training scheme: 'all' (Stage 1 LeJEPA then Stage 2 Decoder), '1' (Stage 1 only), '2' (Stage 2 only), 'baseline'",
    )
    parser.add_argument(
        "--checkpoint_lejepa",
        type=str,
        default=None,
        help="Path to pretrained LeJEPA checkpoint when running Stage 2 standalone",
    )

    # Data & Hyperparameters
    parser.add_argument("--data_path", type=str, default="wiki_train.bin", help="Path to training binary token file")
    parser.add_argument("--val_data_path", type=str, default="wiki_val.bin", help="Path to validation binary token file")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length (context window)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=0, help="Max total steps to train (0 = full epoch; >0 for test runs)")
    parser.add_argument("--stage1_steps", type=int, default=0, help="Max steps for Stage 1 (0 = use max_steps or full epoch)")
    parser.add_argument("--stage2_steps", type=int, default=0, help="Max steps for Stage 2 (0 = use max_steps or full epoch)")

    # Evaluation, Metrics & Logging
    parser.add_argument("--eval_interval", type=int, default=50, help="Steps between validation evaluations")
    parser.add_argument("--eval_steps", type=int, default=10, help="Number of validation batches to evaluate")
    parser.add_argument("--checkpoint_interval", type=int, default=2000, help="Steps between intermediate checkpoint saves")
    parser.add_argument("--save_dir", type=str, default="checkpoints_llm", help="Directory to save checkpoints and metrics")
    parser.add_argument("--prompt", type=str, default="The history of science shows that", help="Prompt for generation test")
    parser.add_argument("--quick_test", action="store_true", help="Run a lightning-fast 10-step sanity check test run")
    parser.add_argument("--no_plot", action="store_true", help="Skip automatic plot generation at completion")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cuda' or 'cpu')",
    )

    return parser.parse_args()


def print_banner(text: str):
    width = 75
    print("\n" + "=" * width)
    print(f"{text.center(width)}")
    print("=" * width + "\n")


def display_model_summary(model, model_type: str):
    print_banner(f"MODEL ARCHITECTURE SUMMARY: {model_type.upper()}")
    param_info = model.count_parameters()
    if model_type == "lejepa":
        print(f"  * LeJEPA Planner (Encoder + Predictor) Params: {param_info['jepa_params']:,} ({param_info['jepa_pct']:.2f}%)")
        print(f"  * Conditional Decoder Params:                 {param_info['decoder_params']:,} ({param_info['decoder_pct']:.2f}%)")
        print(f"  * Total Unique Parameters:                     {param_info['total_params']:,}")
        print("  * Target Parameter Allocation:                ~90% LeJEPA / ~10% Decoder")
        if param_info['jepa_pct'] >= 85.0:
            print("  -> Status: PASSED (Planner carries ~90% of model capacity)")
    else:
        print(f"  * Baseline Transformer LM Total Params:        {param_info['total_params']:,}")
    print("-" * 75)


def save_metrics_to_json(metrics: Dict, save_dir: str, filename: str = "training_metrics.json"):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, filename)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)


@torch.no_grad()
def evaluate_lejepa(model: LeJEPALanguageModel, val_loader: Optional[DataLoader], device: str, max_steps: int = 10):
    if val_loader is None:
        return {}
    model.eval()
    total_ce_loss = 0.0
    total_pred_loss = 0.0
    total_sigreg_loss = 0.0
    count = 0

    for step, batch in enumerate(val_loader):
        if step >= max_steps:
            break
        tokens = batch.to(device)
        targets = tokens[:, 1:]
        tokens_in = tokens[:, :-1]

        # LeJEPA forward
        z_all, _ = model.encoder(tokens)
        z_ctx = z_all[:, :-1, :]
        z_target = z_all[:, 1:, :].detach()
        z_hat = model.predictor(z_ctx)

        pred_loss = F.mse_loss(z_hat, z_target)
        sigreg_loss = model.sigreg(z_all.transpose(0, 1))

        # Conditional Decoder forward
        ce_loss, _ = model.forward_decoder(tokens_in, z_hat, targets)

        total_ce_loss += ce_loss.item()
        total_pred_loss += pred_loss.item()
        total_sigreg_loss += sigreg_loss.item()
        count += 1

    if count == 0:
        return {}
    avg_ce = total_ce_loss / count
    avg_pred = total_pred_loss / count
    avg_sigreg = total_sigreg_loss / count
    ppl = math.exp(min(avg_ce, 20.0))
    return {
        "val_ce_loss": avg_ce,
        "val_ppl": ppl,
        "val_pred_loss": avg_pred,
        "val_sigreg_loss": avg_sigreg,
    }


@torch.no_grad()
def evaluate_baseline(model: StandardTransformerLM, val_loader: Optional[DataLoader], device: str, max_steps: int = 10):
    if val_loader is None:
        return {}
    model.eval()
    total_loss = 0.0
    count = 0

    for step, batch in enumerate(val_loader):
        if step >= max_steps:
            break
        tokens = batch.to(device)
        out = model(tokens)
        total_loss += out["loss"].item()
        count += 1

    if count == 0:
        return {}
    avg_loss = total_loss / count
    ppl = math.exp(min(avg_loss, 20.0))
    return {"val_ce_loss": avg_loss, "val_ppl": ppl}


def train_stage1_lejepa(
    model: LeJEPALanguageModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    metrics_tracker: Dict,
):
    print_banner("STAGE 1: LEJEPA CONTINUOUS PLANNER PRETRAINING (Strict JEPA)")
    print("Objective: Train LeJEPA with L2 Latent Prediction Loss + SIGReg")
    print(f"Regularization: SIGReg lambda = {args.sigreg_weight}")
    print(f"Device: {args.device}\n")

    device = args.device
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    max_steps = args.stage1_steps if args.stage1_steps > 0 else (args.max_steps if args.max_steps > 0 else len(train_loader) * args.epochs)
    global_step = 0
    start_time = time.time()

    s1_metrics = metrics_tracker["stage1"]

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            tokens = batch.to(device)
            optimizer.zero_grad()

            total_loss, pred_loss, sigreg_loss, _, _ = model.forward_lejepa(tokens)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1

            s1_metrics["train_steps"].append(global_step)
            s1_metrics["total_loss"].append(float(total_loss.item()))
            s1_metrics["pred_loss"].append(float(pred_loss.item()))
            s1_metrics["sigreg_loss"].append(float(sigreg_loss.item()))

            if global_step % 10 == 0 or global_step == max_steps:
                elapsed = time.time() - start_time
                print(
                    f"[Stage 1 | Step {global_step}/{max_steps}] "
                    f"Total: {total_loss.item():.4f} | "
                    f"L2 Pred: {pred_loss.item():.4f} | "
                    f"SIGReg: {sigreg_loss.item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            if global_step % args.eval_interval == 0 or global_step == max_steps:
                val_metrics = evaluate_lejepa(model, val_loader, device, max_steps=args.eval_steps)
                if val_metrics:
                    s1_metrics["val_steps"].append(global_step)
                    s1_metrics["val_pred_loss"].append(float(val_metrics['val_pred_loss']))
                    s1_metrics["val_sigreg_loss"].append(float(val_metrics['val_sigreg_loss']))
                    print(
                        f"  -> Validation | L2 Pred: {val_metrics['val_pred_loss']:.4f} | "
                        f"SIGReg: {val_metrics['val_sigreg_loss']:.4f}"
                    )
                # Flush metrics to disk periodically
                save_metrics_to_json(metrics_tracker, args.save_dir)
                model.train()

            if args.checkpoint_interval > 0 and global_step % args.checkpoint_interval == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, "lejepa_stage1_checkpoint.pt")
                torch.save({
                    "encoder_state": model.encoder.state_dict(),
                    "predictor_state": model.predictor.state_dict(),
                    "step": global_step,
                }, ckpt_path)

        if global_step >= max_steps:
            break

    # Save Stage 1 checkpoint & metrics
    os.makedirs(args.save_dir, exist_ok=True)
    stage1_path = os.path.join(args.save_dir, "lejepa_stage1.pt")
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "predictor_state": model.predictor.state_dict(),
            "args": vars(args),
        },
        stage1_path,
    )
    save_metrics_to_json(metrics_tracker, args.save_dir)
    print(f"\n[Stage 1 Complete] LeJEPA weights saved to: {stage1_path}\n")


def train_stage2_decoder(
    model: LeJEPALanguageModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    metrics_tracker: Dict,
):
    print_banner("STAGE 2: CONDITIONAL DECODER TRAINING P(x_{t+1} | x_{\le t}, z_hat_{t+1})")
    print("Scheme A: Freezing LeJEPA Planner (or stop-gradient on z_hat)")
    print("The decoder operates as a true conditional reader on isotropic continuous latents.")
    print(f"Device: {args.device}\n")

    device = args.device
    model.to(device)

    model.freeze_lejepa(freeze=True)
    model.decoder.train()

    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    max_steps = args.stage2_steps if args.stage2_steps > 0 else (args.max_steps if args.max_steps > 0 else len(train_loader) * args.epochs)
    global_step = 0
    start_time = time.time()

    s2_metrics = metrics_tracker["stage2"]

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            tokens = batch.to(device)
            targets = tokens[:, 1:]
            tokens_in = tokens[:, :-1]

            optimizer.zero_grad()

            with torch.no_grad():
                z_all, _ = model.encoder(tokens)
                z_ctx = z_all[:, :-1, :]
                z_hat = model.predictor(z_ctx)

            ce_loss, logits = model.forward_decoder(tokens_in, z_hat, targets)
            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            optimizer.step()

            global_step += 1

            ppl = math.exp(min(ce_loss.item(), 20.0))
            s2_metrics["train_steps"].append(global_step)
            s2_metrics["ce_loss"].append(float(ce_loss.item()))
            s2_metrics["train_ppl"].append(float(ppl))

            if global_step % 10 == 0 or global_step == max_steps:
                elapsed = time.time() - start_time
                print(
                    f"[Stage 2 | Step {global_step}/{max_steps}] "
                    f"CE Loss: {ce_loss.item():.4f} | "
                    f"Train PPL: {ppl:.2f} | "
                    f"Time: {elapsed:.1f}s"
                )

            if global_step % args.eval_interval == 0 or global_step == max_steps:
                val_metrics = evaluate_lejepa(model, val_loader, device, max_steps=args.eval_steps)
                if val_metrics:
                    s2_metrics["val_steps"].append(global_step)
                    s2_metrics["val_ce_loss"].append(float(val_metrics['val_ce_loss']))
                    s2_metrics["val_ppl"].append(float(val_metrics['val_ppl']))
                    print(
                        f"  -> Validation | CE Loss: {val_metrics['val_ce_loss']:.4f} | "
                        f"Val PPL: {val_metrics['val_ppl']:.2f}"
                    )
                save_metrics_to_json(metrics_tracker, args.save_dir)
                model.decoder.train()

            if args.checkpoint_interval > 0 and global_step % args.checkpoint_interval == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, "lejepa_stage2_checkpoint.pt")
                torch.save(model.state_dict(), ckpt_path)

        if global_step >= max_steps:
            break

    # Save full model checkpoint & metrics
    os.makedirs(args.save_dir, exist_ok=True)
    full_path = os.path.join(args.save_dir, "lejepa_full_model.pt")
    torch.save(model.state_dict(), full_path)
    save_metrics_to_json(metrics_tracker, args.save_dir)
    print(f"\n[Stage 2 Complete] Full LeJEPA LM saved to: {full_path}\n")


def train_baseline(
    model: StandardTransformerLM,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    metrics_tracker: Dict,
):
    print_banner("BASELINE STANDARD TRANSFORMER TRAINING P(x_{t+1} | x_{\le t})")
    print("Direct autoregressive next-token prediction without latent planning.")
    print(f"Device: {args.device}\n")

    device = args.device
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    max_steps = args.max_steps if args.max_steps > 0 else len(train_loader) * args.epochs
    global_step = 0
    start_time = time.time()

    base_metrics = metrics_tracker["baseline"]

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            tokens = batch.to(device)
            optimizer.zero_grad()
            res = model(tokens)
            loss = res["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1

            ppl = math.exp(min(loss.item(), 20.0))
            base_metrics["train_steps"].append(global_step)
            base_metrics["ce_loss"].append(float(loss.item()))
            base_metrics["train_ppl"].append(float(ppl))

            if global_step % 10 == 0 or global_step == max_steps:
                elapsed = time.time() - start_time
                print(
                    f"[Baseline | Step {global_step}/{max_steps}] "
                    f"CE Loss: {loss.item():.4f} | "
                    f"Train PPL: {ppl:.2f} | "
                    f"Time: {elapsed:.1f}s"
                )

            if global_step % args.eval_interval == 0 or global_step == max_steps:
                val_metrics = evaluate_baseline(model, val_loader, device, max_steps=args.eval_steps)
                if val_metrics:
                    base_metrics["val_steps"].append(global_step)
                    base_metrics["val_ce_loss"].append(float(val_metrics['val_ce_loss']))
                    base_metrics["val_ppl"].append(float(val_metrics['val_ppl']))
                    print(
                        f"  -> Validation | CE Loss: {val_metrics['val_ce_loss']:.4f} | "
                        f"Val PPL: {val_metrics['val_ppl']:.2f}"
                    )
                save_metrics_to_json(metrics_tracker, args.save_dir)
                model.train()

        if global_step >= max_steps:
            break

    os.makedirs(args.save_dir, exist_ok=True)
    baseline_path = os.path.join(args.save_dir, "baseline_model.pt")
    torch.save(model.state_dict(), baseline_path)
    save_metrics_to_json(metrics_tracker, args.save_dir)
    print(f"\n[Baseline Complete] Baseline model saved to: {baseline_path}\n")


def test_generation(model, tokenizer, prompt: str, device: str, model_name: str):
    print_banner(f"SAMPLE TEXT GENERATION ({model_name.upper()})")
    print(f"Prompt: \"{prompt}\"")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    start_t = time.time()
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=30,
        temperature=0.8,
        top_k=40,
    )
    gen_time = time.time() - start_t
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    print(f"\nGenerated Output:\n{generated_text}")
    print(f"\n[Generation completed in {gen_time:.2f}s]")
    print("-" * 75)


def main():
    args = parse_args()

    # Fast sanity-check test configuration
    if args.quick_test:
        args.batch_size = 4
        args.seq_len = 64
        args.max_steps = 10
        args.eval_interval = 5
        args.eval_steps = 2
        print("\n>>> FAST TEST RUN ENABLED (--quick_test) <<<")
        print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}, Max steps: {args.max_steps}\n")

    # Locate datasets (search local or parent dir)
    for p in [args.data_path, os.path.join("..", args.data_path), os.path.abspath(args.data_path)]:
        if os.path.exists(p):
            args.data_path = p
            break
    for p in [args.val_data_path, os.path.join("..", args.val_data_path), os.path.abspath(args.val_data_path)]:
        if os.path.exists(p):
            args.val_data_path = p
            break

    print(f"Using Training Dataset:   {args.data_path}")
    print(f"Using Validation Dataset: {args.val_data_path}")

    train_loader, val_loader = get_dataloaders(
        train_path=args.data_path,
        val_path=args.val_data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Initialize metrics structure (loads existing if present in save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_file = os.path.join(args.save_dir, "training_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                metrics_tracker = json.load(f)
        except Exception:
            metrics_tracker = {}
    else:
        metrics_tracker = {}

    metrics_tracker.setdefault("stage1", {
        "train_steps": [], "pred_loss": [], "sigreg_loss": [], "total_loss": [],
        "val_steps": [], "val_pred_loss": [], "val_sigreg_loss": [],
    })
    metrics_tracker.setdefault("stage2", {
        "train_steps": [], "ce_loss": [], "train_ppl": [],
        "val_steps": [], "val_ce_loss": [], "val_ppl": [],
    })
    metrics_tracker.setdefault("baseline", {
        "train_steps": [], "ce_loss": [], "train_ppl": [],
        "val_steps": [], "val_ce_loss": [], "val_ppl": [],
    })

    # Mode: Baseline Standard Transformer
    if args.model == "baseline" or args.stage == "baseline":
        model = StandardTransformerLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.baseline_layers,
            dropout=args.dropout,
        )
        display_model_summary(model, "baseline")
        train_baseline(model, train_loader, val_loader, args, metrics_tracker)
        test_generation(model, tokenizer, args.prompt, args.device, "Standard Baseline Transformer")
        if not args.no_plot:
            plot_path = os.path.join(args.save_dir, "training_curves.png")
            plot_training_metrics(metrics_tracker, save_path=plot_path)
        return

    # Mode: Two-Stage Hierarchical LeJEPA Language Model
    model = LeJEPALanguageModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_latent=args.d_latent,
        n_heads=args.n_heads,
        lejepa_encoder_layers=args.lejepa_encoder_layers,
        lejepa_predictor_layers=args.lejepa_predictor_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        sigreg_weight=args.sigreg_weight,
    )
    display_model_summary(model, "lejepa")

    if args.stage in ("1", "all"):
        train_stage1_lejepa(model, train_loader, val_loader, args, metrics_tracker)

    if args.stage == "2":
        if args.checkpoint_lejepa and os.path.exists(args.checkpoint_lejepa):
            print(f"Loading LeJEPA weights from: {args.checkpoint_lejepa}")
            ckpt = torch.load(args.checkpoint_lejepa, map_location=args.device)
            model.encoder.load_state_dict(ckpt["encoder_state"])
            model.predictor.load_state_dict(ckpt["predictor_state"])

    if args.stage in ("2", "all"):
        train_stage2_decoder(model, train_loader, val_loader, args, metrics_tracker)

    # Generation Test
    test_generation(model, tokenizer, args.prompt, args.device, "LeJEPA Hierarchical LM")

    # Auto-generate high-res training curves plot
    if not args.no_plot:
        plot_path = os.path.join(args.save_dir, "training_curves.png")
        plot_training_metrics(metrics_tracker, save_path=plot_path)


if __name__ == "__main__":
    main()
