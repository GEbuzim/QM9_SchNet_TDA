from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
import torch
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader

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
##############
#  3. Нормализация таргета (TRAIN ONLY)
y_train = torch.tensor(
    [data.y[0, TARGET_IDX].item() for data in train_list],
    dtype=torch.float
)

y_mean = y_train.mean()
y_std  = y_train.std()

print(f"Target mean (train): {y_mean:.4f} eV")
print(f"Target std  (train): {y_std:.4f} eV")

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
        geom_out = self.schnet(z, pos, batch)

        # 2. Обрабатываем TDA фичи
        topo_out = self.tda_fc(tda_feat)

        # 3. Склеиваем и выдаем финальное число
        combined = torch.cat([geom_out, topo_out], dim=1)
        return self.final_fc(combined)


# Инициализируем новую модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_schnet = SchNet(hidden_channels=128, num_filters=128, num_interactions=6, cutoff=5.0)
model = TopoSchNet(base_schnet).to(device)
#################################################
# Добавляем индекс в каждый объект данных
for i, data in enumerate(train_list):
    data.idx = i

for i, data in enumerate(test_list):
    data.idx = i

# Пересоздаем лоадеры, чтобы они подхватили новые атрибуты idx
train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_list,  batch_size=32, shuffle=False)

#####################
# Инициализация модели, оптимизатора
base_schnet = SchNet(hidden_channels=128, num_filters=128, num_interactions=6, cutoff=5.0)
model = TopoSchNet(base_schnet).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
#####################
def train_epoch_tda():
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Извлекаем TDA фичи для батча по индексам
        batch_tda = tda_train[data.idx].to(device)

        output = model(data.z, data.pos, data.batch, batch_tda)

        target = data.y[:, TARGET_IDX].view(-1, 1)
        target_norm = (target - y_mean) / y_std

        loss = criterion(output, target_norm)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def eval_mae_tda(loader, tda_storage):
    model.eval()
    total_error = 0.0
    for data in loader:
        data = data.to(device)
        batch_tda = tda_storage[data.idx].to(device)

        output = model(data.z, data.pos, data.batch, batch_tda)
        target = data.y[:, TARGET_IDX].view(-1, 1)

        pred = output * y_std + y_mean
        total_error += (pred - target).abs().sum().item()
    return total_error / len(loader.dataset)
###########################################
num_epochs = 200
best_tda_mae = float("inf")

# лог для TDA модели
log_tda = {
    "epoch": [],
    "train_loss": [],
    "ood_mae": []
}

print("Starting Training with TDA features...\n")

for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch_tda()
    ood_mae = eval_mae_tda(test_loader, tda_test)

    log_tda["epoch"].append(epoch)
    log_tda["train_loss"].append(train_loss)
    log_tda["ood_mae"].append(ood_mae)

    print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | OOD MAE: {ood_mae:.4f} eV")

    # Сохраняем ЛУЧШУЮ ГИБРИДНУЮ модель
    if ood_mae < best_tda_mae:
        best_tda_mae = ood_mae
        torch.save(model.state_dict(), "checkpoints/best_tda_hybrid_model.pt")
        print(f"  ⭐️ New best hybrid model saved (MAE: {ood_mae:.4f})")

# Сохраняем ЛОГ гибридной модели
torch.save(log_tda, "results/schnet_tda_hybrid_log.pt")

print("\nTraining Finished!")
print(f"Best OOD MAE with TDA: {best_tda_mae:.4f} eV")


