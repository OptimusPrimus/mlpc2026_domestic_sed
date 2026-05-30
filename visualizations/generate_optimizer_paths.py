import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

# Loss landscape: narrow quadratic valley
def loss(x, y):
    return 0.15 * x**2 + 2.5 * y**2

def grad(x, y):
    return np.array([0.3 * x, 5.0 * y])


def make_path_sgd(theta0, lr=0.32, steps=35, noise_scale=0.18):
    rng = np.random.default_rng(7)
    theta = np.array(theta0, dtype=float)
    path = [theta.copy()]

    for _ in range(steps):
        g = grad(theta[0], theta[1])
        noise = rng.normal(scale=noise_scale, size=2)
        theta = theta - lr * (g )
        path.append(theta.copy())

    return np.array(path)


def make_path_adamw(theta0, lr=0.32, beta1=0.5, beta2=0.999,
                   eps=1e-8, weight_decay=0.01, steps=35):
    theta = np.array(theta0, dtype=float)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    path = [theta.copy()]

    for t in range(1, steps + 1):
        g = grad(theta[0], theta[1])

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        theta = (
            theta
            - lr * m_hat / (np.sqrt(v_hat) + eps)
            - lr * weight_decay * theta
        )

        path.append(theta.copy())

    return np.array(path)


def plot_path(path, title, filename):
    x = np.linspace(-4, 4, 300)
    y = np.linspace(-2.5, 2.5, 300)
    X, Y = np.meshgrid(x, y)
    Z = loss(X, Y)

    plt.figure(figsize=(5.2, 4.0))
    from matplotlib.colors import LinearSegmentedColormap

    red_cmap =LinearSegmentedColormap.from_list(
    "red_orange",
    [
        "#FFEBEE",  # very light red
        "#FFCDD2",
        "#EF9A9A",
        "#D32F2F"   # darker red
    ]
)


    plt.contourf(
        X,
        Y,
        Z,
        levels=30,
        cmap=red_cmap, alpha=0.1
    )

    plt.contour(
        X,
        Y,
        Z,
        levels=30,
        colors="black",
        linewidths=0.3,
        alpha=0.3
    )
    plt.plot(path[:, 0], path[:, 1], marker="o", markersize=3, linewidth=2)
    plt.scatter([0], [0], marker="*", s=140, label="minimum")

    # plt.title(title)
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, bbox_inches="tight")
    plt.close()


theta0 = (-3.5, 2.0)

sgd_path = make_path_sgd(theta0)
adamw_path = make_path_adamw(theta0)

plot_path(sgd_path, "SGD: noisy zig-zag trajectory", "sgd_path.pdf")
plot_path(adamw_path, "AdamW: adaptive smoother trajectory", "adamw_path.pdf")

print("Saved:")
print(out_dir / "sgd_path.pdf")
print(out_dir / "adamw_path.pdf")