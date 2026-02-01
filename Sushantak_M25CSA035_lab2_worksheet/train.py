import torch
import wandb

from utils.grad_flow import plot_gradient_flow
from data.cifar10_loader import get_cifar10_dataloaders
from models.simple_cnn import SimpleCNN
from utils.train_utils import train_one_epoch, evaluate
from utils.weight_update import WeightTracker


def main():
    # ---------------- Device ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- WandB ----------------
    wandb.init(
        project="cifar10-cnn-lab2",
        name="simple-cnn-step5-training",
        config={
            "epochs": 30,
            "batch_size": 128,
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "architecture": "SimpleCNN",
            "dataset": "CIFAR-10"
        }
    )

    config = wandb.config

    # ---------------- Data ----------------
    train_loader, val_loader, _ = get_cifar10_dataloaders(
        batch_size=config.batch_size
    )

    # ---------------- Model ----------------
    model = SimpleCNN(num_classes=10).to(device)
    weight_tracker = WeightTracker(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9
    )

    # ---------------- Training Loop ----------------
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        plot_gradient_flow(model.named_parameters(), epoch + 1)
        weight_tracker.log_weight_updates(model, epoch + 1)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })          

        print(
            f"Epoch [{epoch+1}/{config.epochs}] | "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
