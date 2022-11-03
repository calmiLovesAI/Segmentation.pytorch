from pathlib import Path

import torch
from tqdm import tqdm

from core.loss import cross_entropy
from utils.tools import MeanMetric


def train_loop(cfg, model, train_dataloader, valid_dataloader):
    print("The training hyperparameters are as follows:")
    for k, v in cfg["Train"].items():
        print(f"{k} : {v}")
    device = cfg["device"]
    dataset_name = cfg["Dataset"]["name"]
    start_epoch = cfg["Train"]["start_epoch"]
    epochs = cfg["Train"]["epochs"]
    save_frequency = cfg["Train"]["save_frequency"]
    save_path = cfg["Train"]["save_path"]
    load_weights = cfg["Train"]["load_weights"]
    tensorboard_on = cfg["Train"]["tensorboard_on"]
    input_size = cfg["Train"]["input_size"]
    batch_size = cfg["Train"]["batch_size"]
    initial_learning_rate = cfg["Train"]["learning_rate"]

    optimizer = torch.optim.SGD(params=model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=0.001)

    loss_mean = MeanMetric()

    if load_weights != "":
        print(f"Successfully loaded weights file: {load_weights}!")
        model.load_state_dict(torch.load(load_weights, map_location=device))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        with tqdm(train_dataloader, desc=f"Epoch-{epoch}/{epochs}") as pbar:
            for i, (images, targets) in enumerate(pbar):
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                preds = model(images)
                loss = cross_entropy(preds, targets)
                loss.backward()
                optimizer.step()
                loss_mean.update(loss.item())

                pbar.set_postfix({
                    "loss": f"{loss_mean.result()}",
                })

        loss_mean.reset()

        evaluate_loop(model, valid_dataloader, device)

        if epoch % save_frequency == 0:
            torch.save(model.state_dict(),
                       Path(save_path).joinpath(f"FCN_{dataset_name}_epoch-{epoch}.pth"))

    torch.save(model.state_dict(),
               Path(save_path).joinpath(f"FCN_{dataset_name}_weights.pth"))
    torch.save(model, Path(save_path).joinpath(f"FCN_{dataset_name}_entire_model.pth"))


def calculate_correct_pixel(pred: torch.Tensor, target: torch.Tensor):
    x = torch.argmax(pred, dim=1)
    # filter out background pixels
    x = torch.where(x > 0, x, -1)
    return (x == target).type(torch.float).sum().item()


def evaluate_loop(model, dataloader, device):
    model.eval()
    test_loss = 0.0
    num_correct_pixels = 0
    num_pixels = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            print(f"Progress: {(100 * (i + 1) / num_batches):.0f}%", end="\r")
            images = images.to(device)
            targets = targets.to(device)
            num_pixels += images.size(0) * images.size(2) * images.size(3)
            pred = model(images)
            test_loss += cross_entropy(pred, targets).item()
            num_correct_pixels += calculate_correct_pixel(pred, targets)
    test_loss /= num_batches
    acc = num_correct_pixels / num_pixels
    print(f"\nEvaluate: Loss: {test_loss:8f}, Accuracy: {(100 * acc):0.2f}%")
