import argparse
import json
import math
import os
import time
from typing import Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from llm.dataset import get_dataloaders
from llm.models import LeJEPALanguageModel, StandardTransformerLM
from llm.plot_metrics import plot_comparison_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-to-head comparison between LeJEPA Hierarchical LM and Standard Transformer LM"
    )
    parser.add_argument("--data_path", type=str, default="wiki_train.bin")
    parser.add_argument("--val_data_path", type=str, default="wiki_val.bin")
    parser.add_argument("--steps", type=int, default=50, help="Default steps for each phase")
    parser.add_argument("--stage1_steps", type=int, default=0, help="Stage 1 LeJEPA steps (0 = use --steps)")
    parser.add_argument("--stage2_steps", type=int, default=0, help="Stage 2 Decoder steps (0 = use --steps)")
    parser.add_argument("--baseline_steps", type=int, default=0, help="Baseline model steps (0 = use --steps)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--d_latent", type=int, default=128, help="Latent dimension for LeJEPA")
    parser.add_argument("--sigreg_weight", type=float, default=0.2, help="Weight lambda for SIGReg regularizer")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=10, help="Evaluation batches")
    parser.add_argument("--checkpoint_interval", type=int, default=2000, help="Steps between intermediate checkpoint saves")
    parser.add_argument("--save_dir", type=str, default="checkpoints_llm", help="Directory to save logs and plots")
    parser.add_argument("--prompt", type=str, default="In natural language processing,", help="Prompt for generation")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"{title.center(80)}")
    print("=" * 80 + "\n")


def train_and_eval_lejepa(args, train_loader, val_loader, compare_data: Dict):
    print_header("TRAINING LEJEPA HIERARCHICAL LANGUAGE MODEL (90-10 Split)")
    device = args.device

    model = LeJEPALanguageModel(
        vocab_size=50257,
        d_model=args.d_model,
        d_latent=args.d_latent,
        n_heads=8,
        lejepa_encoder_layers=7,
        lejepa_predictor_layers=2,
        decoder_layers=1,
        sigreg_weight=args.sigreg_weight,
    ).to(device)

    param_info = model.count_parameters()
    print(f"LeJEPA Planner Params:   {param_info['jepa_params']:,} ({param_info['jepa_pct']:.1f}%)")
    print(f"Conditional Dec Params:  {param_info['decoder_params']:,} ({param_info['decoder_pct']:.1f}%)")
    print(f"Total Model Params:      {param_info['total_params']:,}\n")

    s1_steps = args.stage1_steps if args.stage1_steps > 0 else args.steps
    s2_steps = args.stage2_steps if args.stage2_steps > 0 else args.steps

    # Step 1: Pretrain LeJEPA
    print(f">>> Stage 1: Pretraining LeJEPA Planner ({s1_steps} steps) with L2 Latent Loss + SIGReg...")
    opt_jepa = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
    )
    train_iter = iter(train_loader)
    for step in range(s1_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tokens = batch.to(device)
        opt_jepa.zero_grad()
        loss, pred_loss, sigreg_loss, _, _ = model.forward_lejepa(tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_jepa.step()

        compare_data["lejepa_stage1"]["train_steps"].append(step + 1)
        compare_data["lejepa_stage1"]["pred_loss"].append(float(pred_loss.item()))
        compare_data["lejepa_stage1"]["sigreg_loss"].append(float(sigreg_loss.item()))

        if (step + 1) % 10 == 0 or (step + 1) == s1_steps:
            print(f"  [Stage 1 | Step {step+1}/{s1_steps}] L2 Pred: {pred_loss.item():.4f} | SIGReg: {sigreg_loss.item():.4f}")

        if args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, "lejepa_stage1_checkpoint.pt")
            torch.save({
                "encoder_state": model.encoder.state_dict(),
                "predictor_state": model.predictor.state_dict(),
                "step": step + 1,
            }, ckpt_path)

    # Save Stage 1 final checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    s1_final_path = os.path.join(args.save_dir, "lejepa_stage1.pt")
    torch.save({
        "encoder_state": model.encoder.state_dict(),
        "predictor_state": model.predictor.state_dict(),
        "step": s1_steps,
    }, s1_final_path)
    print(f"\n[Stage 1 Complete] LeJEPA weights saved to: {s1_final_path}\n")

    # Step 2: Train Conditional Decoder with Frozen Planner (Scheme A)
    print(f"\n>>> Stage 2: Training Conditional Decoder ({s2_steps} steps) P(x_{{t+1}} | x_{{\\le t}}, z_hat_{{t+1}})...")
    model.freeze_lejepa(True)
    opt_dec = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr)

    final_train_loss = 0.0
    for step in range(s2_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tokens = batch.to(device)
        targets = tokens[:, 1:]
        tokens_in = tokens[:, :-1]

        opt_dec.zero_grad()
        with torch.no_grad():
            z_all, _ = model.encoder(tokens)
            z_hat = model.predictor(z_all[:, :-1, :])

        loss, _ = model.forward_decoder(tokens_in, z_hat, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        opt_dec.step()
        final_train_loss = loss.item()

        ppl = math.exp(min(loss.item(), 20.0))
        compare_data["lejepa_stage2"]["train_steps"].append(step + 1)
        compare_data["lejepa_stage2"]["ce_loss"].append(float(loss.item()))
        compare_data["lejepa_stage2"]["train_ppl"].append(float(ppl))

        if (step + 1) % 10 == 0 or (step + 1) == s2_steps:
            print(f"  [Stage 2 | Step {step+1}/{s2_steps}] CE Loss: {loss.item():.4f} | Train PPL: {ppl:.2f}")

        if args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, "lejepa_stage2_checkpoint.pt")
            torch.save(model.state_dict(), ckpt_path)

    # Save Stage 2 / Full Model checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    full_model_path = os.path.join(args.save_dir, "lejepa_full_model.pt")
    torch.save(model.state_dict(), full_model_path)
    print(f"\n[Stage 2 Complete] Full LeJEPA LM saved to: {full_model_path}\n")

    # Evaluate on Validation Set
    model.eval()
    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step >= args.eval_steps:
                break
            tokens = batch.to(device)
            targets = tokens[:, 1:]
            tokens_in = tokens[:, :-1]

            z_all, _ = model.encoder(tokens)
            z_hat = model.predictor(z_all[:, :-1, :])
            ce_loss, _ = model.forward_decoder(tokens_in, z_hat, targets)
            val_loss += ce_loss.item()
            val_count += 1

    avg_val_loss = val_loss / max(1, val_count)
    val_ppl = math.exp(min(avg_val_loss, 20.0))

    summary = {
        "params": param_info['total_params'],
        "jepa_pct": param_info['jepa_pct'],
        "dec_pct": param_info['decoder_pct'],
        "final_train_loss": final_train_loss,
        "val_loss": avg_val_loss,
        "val_ppl": val_ppl,
    }
    compare_data["summary"]["lejepa"] = summary
    return model, summary


def train_and_eval_baseline(args, train_loader, val_loader, compare_data: Dict):
    print_header("TRAINING BASELINE STANDARD TRANSFORMER LM")
    device = args.device

    model = StandardTransformerLM(
        vocab_size=50257,
        d_model=args.d_model,
        n_heads=8,
        num_layers=10,  # 10 layers matching LeJEPA total depth
    ).to(device)

    total_params = model.count_parameters()["total_params"]
    print(f"Standard Transformer Total Params: {total_params:,}\n")

    b_steps = args.baseline_steps if args.baseline_steps > 0 else args.steps
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_iter = iter(train_loader)
    final_train_loss = 0.0

    for step in range(b_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tokens = batch.to(device)
        opt.zero_grad()
        res = model(tokens)
        loss = res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_train_loss = loss.item()

        ppl = math.exp(min(loss.item(), 20.0))
        compare_data["baseline"]["train_steps"].append(step + 1)
        compare_data["baseline"]["ce_loss"].append(float(loss.item()))
        compare_data["baseline"]["train_ppl"].append(float(ppl))

        if (step + 1) % 10 == 0 or (step + 1) == b_steps:
            print(f"  [Baseline | Step {step+1}/{b_steps}] CE Loss: {loss.item():.4f} | Train PPL: {ppl:.2f}")

    # Evaluate on Validation Set
    model.eval()
    val_loss = 0.0
    val_count = 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step >= args.eval_steps:
                break
            tokens = batch.to(device)
            res = model(tokens)
            val_loss += res["loss"].item()
            val_count += 1

    avg_val_loss = val_loss / max(1, val_count)
    val_ppl = math.exp(min(avg_val_loss, 20.0))

    # Save Baseline model checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    base_model_path = os.path.join(args.save_dir, "baseline_model.pt")
    torch.save(model.state_dict(), base_model_path)
    print(f"\n[Baseline Complete] Baseline model saved to: {base_model_path}\n")

    summary = {
        "params": total_params,
        "final_train_loss": final_train_loss,
        "val_loss": avg_val_loss,
        "val_ppl": val_ppl,
    }
    compare_data["summary"]["baseline"] = summary
    return model, summary


def main():
    args = parse_args()

    # Find datasets
    for p in [args.data_path, os.path.join("..", args.data_path), os.path.abspath(args.data_path)]:
        if os.path.exists(p):
            args.data_path = p
            break
    for p in [args.val_data_path, os.path.join("..", args.val_data_path), os.path.abspath(args.val_data_path)]:
        if os.path.exists(p):
            args.val_data_path = p
            break

    train_loader, val_loader = get_dataloaders(
        train_path=args.data_path,
        val_path=args.val_data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Metrics container
    compare_data = {
        "lejepa_stage1": {"train_steps": [], "pred_loss": [], "sigreg_loss": []},
        "lejepa_stage2": {"train_steps": [], "ce_loss": [], "train_ppl": []},
        "baseline": {"train_steps": [], "ce_loss": [], "train_ppl": []},
        "summary": {},
    }

    # Run comparisons
    m_jepa, res_jepa = train_and_eval_lejepa(args, train_loader, val_loader, compare_data)
    m_base, res_base = train_and_eval_baseline(args, train_loader, val_loader, compare_data)

    # Print Comparison Table
    print_header("HEAD-TO-HEAD COMPARISON RESULTS")
    print(f"{'Metric':<30} | {'LeJEPA Hierarchical LM':<22} | {'Standard Transformer LM':<22}")
    print("-" * 80)
    print(f"{'Architecture Formulation':<30} | {'P(x_t+1 | x_<=t, z_hat)':<22} | {'P(x_t+1 | x_<=t)':<22}")
    print(f"{'Total Parameters':<30} | {res_jepa['params']:<22,} | {res_base['params']:<22,}")
    pct_str = f"{res_jepa['jepa_pct']:.1f}% / {res_jepa['dec_pct']:.1f}%"
    print(f"{'Parameter Budget Breakdown':<30} | {pct_str:<22} | {'100% monolithic':<22}")
    print(f"{'Final Training CE Loss':<30} | {res_jepa['final_train_loss']:<22.4f} | {res_base['final_train_loss']:<22.4f}")
    print(f"{'Validation CE Loss':<30} | {res_jepa['val_loss']:<22.4f} | {res_base['val_loss']:<22.4f}")
    print(f"{'Validation Perplexity':<30} | {res_jepa['val_ppl']:<22.2f} | {res_base['val_ppl']:<22.2f}")
    print("=" * 80)

    # Generation Samples
    print_header("GENERATION QUALITY SAMPLES")
    print(f"Prompt: \"{args.prompt}\"\n")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)

    print(">>> LeJEPA Hierarchical LM Completion:")
    gen_jepa = m_jepa.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=40)
    print(tokenizer.decode(gen_jepa[0].tolist(), skip_special_tokens=True))

    print("\n>>> Standard Transformer LM Completion:")
    gen_base = m_base.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=40)
    print(tokenizer.decode(gen_base[0].tolist(), skip_special_tokens=True))
    print("=" * 80 + "\n")

    # Save metrics JSON & plot
    os.makedirs(args.save_dir, exist_ok=True)
    json_path = os.path.join(args.save_dir, "compare_metrics.json")
    with open(json_path, "w") as f:
        json.dump(compare_data, f, indent=2)
    print(f"[Comparison Complete] Metrics saved to: {json_path}")

    plot_path = os.path.join(args.save_dir, "comparison_curves.png")
    plot_comparison_metrics(compare_data, save_path=plot_path)


if __name__ == "__main__":
    main()
