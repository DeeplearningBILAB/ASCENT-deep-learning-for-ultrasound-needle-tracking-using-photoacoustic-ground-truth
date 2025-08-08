import os
import torch
import config
from tqdm import tqdm
from model.network import AscentPlus
from matplotlib import pyplot as plt
from utils.loader import CustomDataset
from utils.loss import muti_loss_fusion, DiceLoss
from monai.data import (DataLoader, CacheDataset)
import torch.optim.lr_scheduler as schedulers

train_dataset = CustomDataset(config.TRAIN_FILENAME)
train_ds = CacheDataset(train_dataset, transform=config.TRAIN_TRANSFORMS, num_workers=4, cache_rate=0.5)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
val_dataset = CustomDataset(config.VALID_FILENAME)
val_ds = CacheDataset(val_dataset, num_workers=0, cache_rate=0.5)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device(f"cuda:{config.DEVICE_IDX}" if torch.cuda.is_available() else "cpu")
model = AscentPlus(in_channels=3, num_classes=1, output_stride=16, pretrained=True).to(device)

torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = schedulers.CosineAnnealingLR(optimizer, T_max=config.MAX_ITERATIONS, eta_min=1e-6)
dice_calculator = DiceLoss()


def validation(epoch_iterator_val):
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inp = batch["image"].type(torch.FloatTensor)
            val_lbl = batch["label"].type(torch.FloatTensor)
            val_inputs, val_labels = (val_inp.to(device), val_lbl.to(device))
            dv0, dv1, dv2, dv3, dv4, dv5, dv6 = model(val_inputs)
            loss2, loss = muti_loss_fusion(dv0, dv1, dv2, dv3, dv4, dv5, dv6, val_labels)

            pred = (dv0 > 0.5).float()
            intersection = (pred * val_labels).sum()
            dice = (2. * intersection) / (pred.sum() + val_labels.sum() + 1e-5)
            dice_percent = dice.item() * 100

            total_loss += loss.item()
            total_dice += dice_percent
            num_batches += 1

    average_loss = total_loss / num_batches
    average_dice = total_dice / num_batches

    print(f"Validation Accuracy: {average_dice:.2f}%")
    return average_loss, average_dice


if __name__ == "__main__":
    epoch_loss_values = []
    metric_values = []
    dice_val_best = 10.0
    global_step_best = 0
    global_step = config.GLOBAL_STEP
    model.train()

    while global_step < config.MAX_ITERATIONS:
        epoch_loss = 0
        epoch_dice = 0
        epoch_step = 0
        epoch_iterator = tqdm(train_loader, desc=f"Training Iteration {global_step}/{config.MAX_ITERATIONS}",
                              dynamic_ncols=True)

        for batch in epoch_iterator:
            if global_step >= config.MAX_ITERATIONS:
                break

            epoch_step += 1
            x_img = batch["image"].type(torch.FloatTensor)
            y_img = batch["label"].type(torch.FloatTensor)
            x, y = (x_img.to(device), y_img.to(device))

            optimizer.zero_grad()
            d0, d1, d2, d3, d4, d5, d6 = model(x)
            loss2, loss = muti_loss_fusion(d0, d1, d2, d3, d4, d5, d6, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            if global_step < 1000:
                lr = 1e-5 + (global_step / 1000) * (1e-4 - 1e-5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                scheduler.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                pred = (d0 > 0.5).float()
                intersection = (pred * y).sum()
                dice = (2. * intersection) / (pred.sum() + y.sum() + 1e-5)
                dice_percent = dice.item() * 100
                epoch_dice += dice_percent

            epoch_iterator.set_description(
                f"Training ({global_step}/{config.MAX_ITERATIONS}) (loss={loss.item():.5f}, acc={dice_percent:.2f}%)"
            )
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss, pred, x_img, y_img, x, y  # Memory cleanup

            global_step += 1

            if (global_step % config.EVALUATION_NUM == 0 and global_step != 0) or global_step == config.MAX_ITERATIONS:
                model.eval()
                epoch_iterator_val = tqdm(val_loader, desc="Validation", dynamic_ncols=True)
                dice_val, dice_acc = validation(epoch_iterator_val)

                current_epoch_avg_loss = epoch_loss / epoch_step if epoch_step > 0 else 0
                current_epoch_avg_dice = epoch_dice / epoch_step if epoch_step > 0 else 0
                epoch_loss_values.append(current_epoch_avg_loss)
                metric_values.append(dice_val)

                print(
                    f"\nIteration {global_step}: Training Avg Loss: {current_epoch_avg_loss:.4f}, Training Avg Acc: {current_epoch_avg_dice:.2f}%")
                print(f"Iteration {global_step}: Validation Loss: {dice_val:.4f}, Validation Acc: {dice_acc:.2f}%")

                if dice_val < dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, config.MODEL_NAME))
                    print(
                        f"Model Saved! Iteration: {global_step}, Best Val Loss: {dice_val_best:.4f}, Val Acc: {dice_acc:.2f}%"
                    )
                else:
                    print(
                        f"Model Not Saved. Best Val Loss: {dice_val_best:.4f} at iter {global_step_best}, Current Val Loss: {dice_val:.4f}, Val Acc: {dice_acc:.2f}%"
                    )

                model.train()

                epoch_loss = 0
                epoch_dice = 0
                epoch_step = 0

            if global_step >= config.MAX_ITERATIONS:
                break

        if global_step >= config.MAX_ITERATIONS:
            break

    if os.path.exists(os.path.join(config.MODEL_DIR, config.MODEL_NAME)):
        print(f"\nLoading best model from iteration {global_step_best} with loss {dice_val_best:.4f}")
        model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, config.MODEL_NAME)))
    else:
        print("\nNo best model was saved during training.")

    print(f"\nTraining completed after {global_step} iterations.")
    print(f"Best validation loss: {dice_val_best:.4f} at iteration: {global_step_best}")

    plt.figure("Training Results", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Training Loss")
    plot_steps = [config.EVALUATION_NUM * (i + 1) for i in range(len(epoch_loss_values))]
    if global_step % config.EVALUATION_NUM != 0 and global_step == config.MAX_ITERATIONS:
        if not plot_steps or plot_steps[-1] < global_step:
            plot_steps.append(global_step)

    if len(plot_steps) > len(epoch_loss_values):
        plot_steps = plot_steps[:len(epoch_loss_values)]
    elif len(epoch_loss_values) > len(plot_steps):
        epoch_loss_values = epoch_loss_values[:len(plot_steps)]

    plt.plot(plot_steps, epoch_loss_values)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.title("Validation Loss per Evaluation")
    val_plot_steps = [config.EVALUATION_NUM * (i + 1) for i in range(len(metric_values))]
    if global_step % config.EVALUATION_NUM != 0 and global_step == config.MAX_ITERATIONS:
        if not val_plot_steps or val_plot_steps[-1] < global_step:
            val_plot_steps.append(global_step)

    if len(val_plot_steps) > len(metric_values):
        val_plot_steps = val_plot_steps[:len(metric_values)]
    elif len(metric_values) > len(val_plot_steps):
        metric_values = metric_values[:len(val_plot_steps)]

    plt.plot(val_plot_steps, metric_values)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")

    plt.tight_layout()
    plt.show()
