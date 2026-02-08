import os
import torch
import numpy as np

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

torch.manual_seed(42)
np.random.seed(42)
#########################################
#   1. Загрузка датасета и OOD split
dataset = QM9(root="data/QM9")

TARGET_IDX = 4  # HOMO-LUMO gap


def num_heavy_atoms(data):
    return (data.z > 1).sum().item()


train_list, test_list = [], []

for data in dataset:
    if num_heavy_atoms(data) == 9:
        test_list.append(data)
    else:
        train_list.append(data)

print(f"Train size: {len(train_list)}")
print(f"OOD test size (n=9): {len(test_list)}")

######################################################
#   2. DataLoaders
train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_list,  batch_size=32, shuffle=False)
#################################################################
#  3. Нормализация таргета (TRAIN ONLY)
y_train = torch.tensor(
    [data.y[0, TARGET_IDX].item() for data in train_list],
    dtype=torch.float
)

y_mean = y_train.mean()
y_std  = y_train.std()

print(f"Target mean (train): {y_mean:.4f} eV")
print(f"Target std  (train): {y_std:.4f} eV")
################################################
# 3    Сохраняем статистики:

os.makedirs("results", exist_ok=True)
torch.save(
    {"mean": y_mean, "std": y_std},
    "results/target_normalization.pt"
)

###############################################
# 4. Модель, оптимизатор, loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SchNet(
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    num_gaussians=50,
    cutoff=5.0
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
#####################################

#  5. Train / Test функции
def train_epoch():
    model.train()
    total_loss = 0.0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data.z, data.pos, data.batch)

        target = data.y[:, TARGET_IDX].view(-1, 1)
        target_norm = (target - y_mean) / y_std

        loss = criterion(output, target_norm)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def eval_mae(loader):
    model.eval()
    total_error = 0.0

    for data in loader:
        data = data.to(device)

        output = model(data.z, data.pos, data.batch)
        target = data.y[:, TARGET_IDX].view(-1, 1)

        pred = output * y_std + y_mean
        total_error += (pred - target).abs().sum().item()

    return total_error / len(loader.dataset)
###################################################
#  6. Основной training loop с логированием
num_epochs = 200
best_ood_mae = float("inf")

log = {
    "epoch": [],
    "train_loss": [],
    "ood_mae": []
}

os.makedirs("checkpoints", exist_ok=True)

print("Starting training...\n")

for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch()
    ood_mae = eval_mae(test_loader)

    log["epoch"].append(epoch)
    log["train_loss"].append(train_loss)
    log["ood_mae"].append(ood_mae)

    print(
        f"Epoch {epoch:03d} | "
        f"Train MSE: {train_loss:.4f} | "
        f"OOD MAE: {ood_mae:.4f} eV"
    )

    # сохраняем лучшую модель
    if ood_mae < best_ood_mae:
        best_ood_mae = ood_mae
        torch.save(
            model.state_dict(),
            "checkpoints/best_schnet_normalized.pt"
        )
        print("  ✔ New best model saved")

# сохраняем лог
torch.save(log, "results/schnet_normalized_log.pt")

print("\nTraining finished.")
print(f"Best OOD MAE (n=5): {best_ood_mae:.4f} eV")
##########################################################