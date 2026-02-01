import torch
import matplotlib.pyplot as plt
import wandb


class WeightTracker:
    def __init__(self, model):
        self.prev_weights = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def log_weight_updates(self, model, epoch):
        update_ratios = []
        layers = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                prev = self.prev_weights[name]
                curr = param.detach()

                delta = torch.norm(curr - prev)
                norm = torch.norm(prev) + 1e-8

                update_ratios.append((delta / norm).item())
                layers.append(name)

                # update stored weights
                self.prev_weights[name] = curr.clone()

        fig = plt.figure(figsize=(10, 5))
        plt.plot(update_ratios, alpha=0.8)
        plt.xticks(range(len(layers)), layers, rotation="vertical")
        plt.xlabel("Layers")
        plt.ylabel("||ΔW|| / ||W||")
        plt.title(f"Weight Update Flow at Epoch {epoch}")
        plt.tight_layout()

        wandb.log({
            "Weight Update Flow": wandb.Image(fig),
            "epoch": epoch
        })

        plt.close(fig)
