from pathlib import Path

import torch
from tqdm import tqdm

from core.loss import cross_entropy
from utils.tools import MeanMetric


def train_loop(cfg, model, train_loader, valid_loader):
    print("The training hyperparameters are as follows:")
    for k, v in cfg["Train"].items():
        print(f"{k} : {v}")
    device = cfg["device"]
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
        with tqdm(train_loader, desc=f"Epoch-{epoch}/{epochs}") as pbar:
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

        if epoch % save_frequency == 0:
            torch.save(model.state_dict(),
                       Path(save_path).joinpath("FCN_{}_epoch-{}.pth".format(cfg["dataset"], epoch)))

    torch.save(model.state_dict(),
               Path(save_path).joinpath("FCN_{}_weights.pth".format(cfg["dataset"])))
    torch.save(model, Path(save_path).joinpath("FCN_{}_entire_model.pth".format(cfg["dataset"])))