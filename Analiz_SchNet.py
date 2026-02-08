import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка лучшей модели
model = SchNet(
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    num_gaussians=50,
    cutoff=5.0
)

model.load_state_dict(
    torch.load("checkpoints/best_schnet_normalized.pt",
               map_location=device)
)

model = model.to(device)
model.eval()

#   1. Загрузка датасета и OOD split
dataset = QM9(root="data/QM9")

TARGET_IDX = 4  # HOMO-LUMO gap



#############################################################
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
#############################################################
#  3. Нормализация таргета (TRAIN ONLY)
y_train = torch.tensor(
    [data.y[0, TARGET_IDX].item() for data in train_list],
    dtype=torch.float
)

y_mean = y_train.mean()
y_std  = y_train.std()

print(f"Target mean (train): {y_mean:.4f} eV")
print(f"Target std  (train): {y_std:.4f} eV")

#################################  График Predicted vs True
@torch.no_grad()
def predict_phys(data):
    data = data.to(device)
    pred_norm = model(
        data.z,
        data.pos, data.batch)

    pred_phys = pred_norm.item() * y_std + y_mean
    return pred_phys

#################
true_vals = []
pred_vals = []

for data in tqdm(test_list):  # OOD: n=5
    pred = predict_phys(data)
    true = data.y[0, TARGET_IDX].item()

    pred_vals.append(pred)
    true_vals.append(true)

plt.figure(figsize=(6,6))
plt.scatter(true_vals, pred_vals, s=10, alpha=0.6)
plt.plot(
    [min(true_vals), max(true_vals)],
    [min(true_vals), max(true_vals)],
    "r--"
)
plt.xlabel("True HOMO-LUMO gap (eV)")
plt.ylabel("Predicted HOMO-LUMO gap (eV)")
plt.title("OOD (n=9): Prediction vs Ground Truth")
plt.grid(True)
#plt.show()
plt.savefig(f'Pred_VS_Truth.png')





