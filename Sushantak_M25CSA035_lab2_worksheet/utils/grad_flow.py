import matplotlib.pyplot as plt
import wandb


def plot_gradient_flow(named_parameters, epoch):
    ave_grads = []
    layers = []

    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())

    fig = plt.figure(figsize=(10, 5))
    plt.plot(ave_grads, alpha=0.8)
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(len(layers)), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average |Gradient|")
    plt.title(f"Gradient Flow at Epoch {epoch}")
    plt.tight_layout()

    wandb.log({
        "Gradient Flow": wandb.Image(fig),
        "epoch": epoch
    })

    plt.close(fig)
