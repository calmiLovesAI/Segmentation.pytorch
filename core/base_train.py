from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.metrics import SegmentationMetrics
from core.optimizer import get_optimizer, get_lr_scheduler
from utils.tools import MeanMetric, Saver, show_cfg


def train_loop(cfg, model, criterion, train_dataloader, valid_dataloader):
    print("The parameters in the configuration file are as follows:")
    show_cfg(cfg)
    device = cfg["device"]
    model_name = cfg["Model"]["name"]
    dataset_name = cfg["Dataset"]["name"]

    epochs = cfg["Train"]["epochs"]
    save_frequency = cfg["Train"]["save_frequency"]
    save_path = cfg["Train"]["save_path"]
    ckpt_file = cfg["Train"]["load_weights"]
    tensorboard_on = cfg["Train"]["tensorboard_on"]
    input_size = cfg["Train"]["input_size"]
    batch_size = cfg["Train"]["batch_size"]
    initial_learning_rate = cfg["Train"]["learning_rate"]
    optimizer_name = cfg["Train"]["optimizer"]
    scheduler_name = cfg["Train"]["scheduler"]
    milestones = cfg["Train"]["milestones"]
    gamma = cfg["Train"]["gamma"]

    optimizer = get_optimizer(model=model, optimizer_name=optimizer_name, lr=initial_learning_rate)
    scheduler = get_lr_scheduler(optimizer, scheduler_name=scheduler_name, milestones=milestones, gamma=gamma)

    loss_mean = MeanMetric()
    saver = Saver(model, optimizer, scheduler)

    if ckpt_file != "":
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler != "None":
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["current_epoch"] + 1
        saver.set_best_score(best_score=ckpt["best_score"])
        print(f"Successfully loaded checkpoint: {ckpt_file}!")
        del ckpt  # free memory
    else:
        start_epoch = 0

    if tensorboard_on:
        # 在控制台使用命令 tensorboard --logdir=runs 进入tensorboard面板
        writer = SummaryWriter()
        writer.add_graph(model, torch.randn(batch_size, *input_size, dtype=torch.float32, device=device))

    for epoch in range(start_epoch, epochs):
        model.train()
        loss_mean.reset()
        with tqdm(train_dataloader, desc=f"Epoch-{epoch}/{epochs}") as pbar:
            for i, (images, targets) in enumerate(pbar):
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                loss_mean.update(loss.item())

                if tensorboard_on:
                    writer.add_scalar(tag="train/Loss", scalar_value=loss_mean.result(),
                                      global_step=epoch * len(train_dataloader) + i)

                pbar.set_postfix({
                    "loss": f"{loss_mean.result()}",
                })
        if scheduler != "None":
            scheduler.step()

        test_loss, score = evaluate_loop(cfg, model, valid_dataloader)
        if tensorboard_on:
            writer.add_scalar(tag="val/Loss", scalar_value=test_loss, global_step=epoch)
            writer.add_scalar(tag="val/Overall Acc", scalar_value=score["Overall Acc"], global_step=epoch)
            writer.add_scalar(tag="val/Mean Acc", scalar_value=score["Mean Acc"], global_step=epoch)
            writer.add_scalar(tag="val/FreqW Acc", scalar_value=score["FreqW Acc"], global_step=epoch)
            writer.add_scalar(tag="val/Mean IoU", scalar_value=score["Mean IoU"], global_step=epoch)

        if epoch % save_frequency == 0:
            saver.save_ckpt(epoch=epoch,
                            save_root=save_path,
                            filename_prefix=f"{model_name}_{dataset_name}",
                            score=score["Mean IoU"],
                            overwrite=True)

    saver.save_ckpt(epoch=epochs - 1,
                    save_root=save_path,
                    filename_prefix=f"{model_name}_{dataset_name}",
                    score=score["Mean IoU"],
                    overwrite=True)
    torch.save(model.state_dict(), Path(save_path).joinpath(f"{model_name}_{dataset_name}_weights.pth"))

    if tensorboard_on:
        writer.close()


def evaluate_loop(cfg, model, criterion, dataloader):
    device = cfg["device"]
    model.eval()
    test_loss = 0.0

    metrics = SegmentationMetrics(num_classes=cfg["Dataset"]["num_classes"])
    num_batches = len(dataloader)
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            print(f"Progress: {(100 * (i + 1) / num_batches):.0f}%", end="\r")
            images = images.to(device)
            targets = targets.to(device)
            pred = model(images)
            test_loss += criterion(pred, targets).item()
            pred = torch.argmax(pred, dim=1)
            metrics.add_batch(predictions=pred.cpu().numpy(), gts=targets.cpu().numpy())

    test_loss /= num_batches
    metric_results = metrics.get_results()
    print(f"\nEvaluate: Loss: {test_loss:8f}, mIoU: {metric_results['Mean IoU']:0.4f}")
    return test_loss, metric_results
