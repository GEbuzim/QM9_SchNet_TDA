import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################
print("Device:", device)

# Путь к чекпоинтам и нормализации
schnet_ckpt = "checkpoints/best_schnet_normalized.pt"
tda_ckpt   = "checkpoints/best_tda_hybrid_model.pt"
norm_path  = "results/target_normalization.pt"

# Загружаем статистики нормализации (mean, std)
if os.path.exists(norm_path):
    stats = torch.load(norm_path, map_location=device)
    y_mean = stats["mean"].to(device)
    y_std  = stats["std"].to(device)
    print("Loaded normalization stats:", y_mean.item(), y_std.item())
else:
    raise FileNotFoundError(f"Normalization stats not found at {norm_path}")

print(f"Target mean (train): {y_mean:.4f} eV")
print(f"Target std  (train): {y_std:.4f} eV")
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

#######################################
# Инициализируем компоненты giotto-tda
# homology_dimensions=[0, 1] означает, что мы смотрим на компоненты связности и кольца
VR = VietorisRipsPersistence(homology_dimensions=[0, 1])
PE = PersistenceEntropy()

def compute_tda_features(data_list):
    # Извлекаем координаты всех молекул
    X = [data.pos.numpy() for data in data_list]

    # 1. Строим диаграммы персистентности
    print("Вычисляем диаграммы персистентности...")
    diagrams = VR.fit_transform(X)

    # 2. Извлекаем энтропию (топологическую сложность)
    print("Вычисляем персистентную энтропию...")
    features = PE.fit_transform(diagrams)

    return torch.tensor(features, dtype=torch.float)


# Считаем для трейна и теста
tda_train = compute_tda_features(train_list)
tda_test = compute_tda_features(test_list)

print(f"Форма признаков TDA: {tda_train.shape}")  # Должно быть [N, 2]

######################################
class TopoSchNet(torch.nn.Module):
    def __init__(self, base_schnet, tda_dim=2):
        super().__init__()
        self.schnet = base_schnet

        # Дополнительный слой для обработки TDA фич
        self.tda_fc = torch.nn.Sequential(
            torch.nn.Linear(tda_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16)
        )

        # Финальная голова, объединяющая Геометрию и Топологию
        self.final_fc = torch.nn.Linear(1 + 16, 1)

    def forward(self, z, pos, batch, tda_feat):
        # 1. Получаем выход SchNet (геометрия)
        geom_out = self.schnet(z, pos, batch)  # [batch, 1]

        # 2. Обрабатываем TDA фичи
        topo_out = self.tda_fc(tda_feat)  # [batch, 16]

        # 3. Склеиваем и выдаем финальное число
        combined = torch.cat([geom_out, topo_out], dim=1)
        return self.final_fc(combined)

#######################################
# Добавляем индекс в каждый объект данных
for i, data in enumerate(train_list):
    data.idx = i

for i, data in enumerate(test_list):
    data.idx = i

# Пересоздаем лоадеры, чтобы они подхватили новые атрибуты idx
train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_list,  batch_size=32, shuffle=False)
##########################################
base_schnet = SchNet(hidden_channels=128, num_filters=128, num_interactions=6, cutoff=5.0)
model_tda = TopoSchNet(base_schnet).to(device)

###################################################
# Загружаем веса
if os.path.exists(tda_ckpt):
    state = torch.load(tda_ckpt, map_location=device)
    try:
        model_tda.load_state_dict(state)
        print("TopoSchNet weights loaded successfully.")
    except RuntimeError as e:
        print("Strict load failed:", e)
        model_tda.load_state_dict(state, strict=False)
        print("Loaded with strict=False (check for missing keys).")
else:
    raise FileNotFoundError(f"TDA hybrid checkpoint not found at {tda_ckpt}")
#######################################################
#################################  График  Predicted vs True
model_tda.eval()
@torch.no_grad()
def predict_phys(data):
    data = data.to(device)
    idx = int(data.idx)  # индекс в tda_test / tda_train
    batch_tda = tda_test[idx].unsqueeze(0).to(device)
    pred_norm = model_tda(
        data.z,
        data.pos, data.batch, batch_tda)
    pred_phys = (pred_norm * y_std + y_mean).squeeze().cpu().item()
    return pred_phys

#################
true_vals = []
pred_vals = []

for data in tqdm(test_list):
    pred = predict_phys(data)
    true = data.y[0, TARGET_IDX].item()

    pred_vals.append(pred)
    true_vals.append(true)

print(len(true_vals),len(pred_vals))
for i in range(10):
    print(i,true_vals[i],pred_vals[i])

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
plt.savefig(f'Pred_VS_Truth_SchNet_TDA_9.png')




#############################


