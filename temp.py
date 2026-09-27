import torch
from torch.utils.data import DataLoader
# pyrefly: ignore [missing-import]
from module import SIGReg

def train_lejepa_step_by_step(
    model,                  # instance of JEPA (from jepa.py)
    train_dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    num_epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    ctx_len: int = 3,       # cfg.history_size: number of context frames
    n_preds: int = 1,       # cfg.num_preds: target shift/offset
    lambd: float = 0.1,     # cfg.loss.sigreg.weight: regularization weight
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    
    # 1. Regularizer (SIGReg: Sketch Isotropic Gaussian Regularizer to prevent collapse)
    sigreg = SIGReg().to(device)

    # 2. Optimizer & Learning Rate Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # -----------------------------
    # Epoch Loop
    # -----------------------------
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_pred_loss = 0.0
        total_sigreg_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Move data to target device
            pixels = batch["pixels"].to(device) # Shape: (B, T, C, H, W)
            actions = batch["action"].to(device) # Shape: (B, T, act_dim)

            # Clean NaNs at sequence edges
            actions = torch.nan_to_num(actions, 0.0)

            # ==========================================
            # STEP 1: Encode observations and actions
            # ==========================================
            # model.encode extracts CLS tokens through the visual encoder 
            # and actions through action_encoder
            encoded_info = model.encode({"pixels": pixels, "action": actions})
            emb = encoded_info["emb"]         # (B, T, D)
            act_emb = encoded_info["act_emb"] # (B, T, act_emb_dim)

            # ==========================================
            # STEP 2: Slice Context and Targets
            # ==========================================
            # Past context given to the predictor
            ctx_emb = emb[:, :ctx_len]        # (B, ctx_len, D)
            ctx_act = act_emb[:, :ctx_len]    # (B, ctx_len, act_emb_dim)

            # Future target ground-truth embeddings in latent space
            tgt_emb = emb[:, n_preds:]        # (B, T - n_preds, D)

            # ==========================================
            # STEP 3: Predict Future Latent States
            # ==========================================
            # model.predict uses ARPredictor (Transformer + AdaLN)
            pred_emb = model.predict(ctx_emb, ctx_act) # (B, ctx_len, D)

            # ==========================================
            # STEP 4: Compute Losses
            # ==========================================
            # 4a. Prediction loss in latent space (L2 / MSE loss)
            pred_loss = (pred_emb - tgt_emb).pow(2).mean()

            # 4b. SIGReg Regularization loss
            # sigreg expects shape (T, B, D) to prevent dimensional/representation collapse
            sigreg_loss = sigreg(emb.transpose(0, 1))

            # 4c. Total Loss
            loss = pred_loss + lambd * sigreg_loss

            # ==========================================
            # STEP 5: Backpropagation & Optimization
            # ==========================================
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional stability
            optimizer.step()

            # Logging metrics
            total_train_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_sigreg_loss += sigreg_loss.item()

            if step % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{len(train_dataloader)}] "
                    f"Loss: {loss.item():.4f} (Pred: {pred_loss.item():.4f}, SIGReg: {sigreg_loss.item():.4f})"
                )

        scheduler.step()

        # ==========================================
        # STEP 6: Validation Loop (Evaluation)
        # ==========================================
        if val_dataloader is not None:
            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    pixels = val_batch["pixels"].to(device)
                    actions = torch.nan_to_num(val_batch["action"].to(device), 0.0)

                    encoded_info = model.encode({"pixels": pixels, "action": actions})
                    emb = encoded_info["emb"]
                    act_emb = encoded_info["act_emb"]

                    ctx_emb = emb[:, :ctx_len]
                    ctx_act = act_emb[:, :ctx_len]
                    tgt_emb = emb[:, n_preds:]

                    pred_emb = model.predict(ctx_emb, ctx_act)
                    val_pred_loss = (pred_emb - tgt_emb).pow(2).mean()
                    val_sigreg_loss = sigreg(emb.transpose(0, 1))
                    val_loss = val_pred_loss + lambd * val_sigreg_loss
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"--> Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")
