import numpy as np
import matplotlib.pyplot as plt
fs = 14
# Global font sizes
plt.rcParams.update({
    "font.size": fs,
    "axes.labelsize": fs,
    "xtick.labelsize": fs,
    "ytick.labelsize": fs,
    "legend.fontsize": fs,
})

# Parameters
epochs = 50
start_decay_epoch = 20
initial_lr = 1e-3
min_lr = 1e-5

# Compute learning rate schedule
lr = []

for epoch in range(epochs):
    if epoch < start_decay_epoch:
        current_lr = initial_lr
    else:
        progress = (epoch - start_decay_epoch) / (epochs - 1 - start_decay_epoch)
        current_lr = initial_lr - (initial_lr - min_lr) * progress

    lr.append(current_lr)

# Plot
plt.figure(figsize=(8, 2.5))
plt.plot(range(epochs), lr, linewidth=3)

# Mark important points
plt.axvline(start_decay_epoch, linestyle="--", alpha=0.7)
plt.axhline(initial_lr, linestyle=":", alpha=0.7)
plt.axhline(min_lr, linestyle=":", alpha=0.7)

plt.annotate(
    r"$\eta_0 = 10^{-3}$",
    xy=(2, initial_lr),
    xytext=(5, initial_lr * 1.1),
    fontsize=14,
)

plt.annotate(
    r"$\eta_{\min} = 10^{-5}$",
    xy=(epochs - 5, min_lr),
    xytext=(30, min_lr + 8e-5),
    fontsize=fs,
)


plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.xlim(0, epochs - 1)

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("lr_scheduler.pdf", bbox_inches="tight")
plt.show()