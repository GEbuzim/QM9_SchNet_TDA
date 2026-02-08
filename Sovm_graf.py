import torch
import matplotlib.pyplot as plt

# Загружаем логи (пути к логам)
log_base = torch.load("results/schnet_normalized_log.pt")
log_tda  = torch.load("results/schnet_tda_hybrid_log.pt")

plt.figure(figsize=(8,4))
plt.plot(log_base["epoch"], log_base["ood_mae"], label="SchNet (base)", color='orange', alpha=0.9)
plt.plot(log_tda["epoch"],  log_tda["ood_mae"],  label="SchNet + TDA", color='royalblue', alpha=0.9)
plt.axhline(y=min(log_base["ood_mae"]), color='orange', linestyle='--', alpha=0.6, label=f'Best base: {min(log_base["ood_mae"]):.3f}')
plt.axhline(y=min(log_tda["ood_mae"]),  color='royalblue', linestyle='--', alpha=0.6, label=f'Best TDA: {min(log_tda["ood_mae"]):.3f}')
plt.xlabel("Epoch")
plt.ylabel("OOD MAE (eV)")
plt.title("SchNet vs SchNet+TDA — OOD MAE")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/ood_mae_comparison_9.png", dpi=200)
#plt.show()