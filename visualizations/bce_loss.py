import numpy as np
import matplotlib.pyplot as plt

# Predicted probabilities
p = np.linspace(1e-6, 1 - 1e-6, 1000)

# BCE loss for the two possible labels
loss_y1 = -np.log(p)          # y = 1
loss_y0 = -np.log(1 - p)      # y = 0

plt.figure(figsize=(4, 4))

plt.plot(p, loss_y1, label=r"$y=1:\;-\log(\hat y)$")
plt.plot(p, loss_y0, label=r"$y=0:\;-\log(1-\hat y)$")

plt.xlabel(r"Predicted Probability $\hat y$")
plt.ylabel("Binary Cross Entropy Loss")
#plt.title("Binary Cross Entropy Loss for a Single Prediction")
plt.xlim(0, 1)
plt.ylim(0, 8)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("bce_loss.pdf", bbox_inches="tight")
plt.show()