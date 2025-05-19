# src/training.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from config import RANDOM_SEED, MODEL_DIR, SCHEDULER_PARAMS


def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    torch.manual_seed(RANDOM_SEED)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, **SCHEDULER_PARAMS)
    scaler = GradScaler()

    best_auc, epochs_no_improve = 0, 0
    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            with autocast():
                preds = model(xb.float())
                loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, yb.float().unsqueeze(1), weight=torch.tensor(class_weights))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
    torch.save(model.state_dict(), MODEL_DIR/'best_model.pt')
