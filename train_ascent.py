import os
import sys
import argparse
import csv
import torch
import config
from tqdm import tqdm
from matplotlib import pyplot as plt
from network.ascent_plus import AscentPlus
from utils.loader import CustomDataset
from utils.loss import muti_loss_fusion
from utils.train_utils import dice_iou_from_logits
from monai.data import DataLoader, CacheDataset
import torch.optim.lr_scheduler as schedulers


def parse_args():
    p = argparse.ArgumentParser(description="Train ASCENT+")
    p.add_argument("--use_contrast_boost", action="store_true", default=True)
    p.add_argument("--no_contrast_boost", action="store_false", dest="use_contrast_boost")
    p.add_argument("--use_spam", action="store_true", default=True)
    p.add_argument("--no_spam", action="store_false", dest="use_spam")
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101", "mobilenetv3"])
    p.add_argument("--loss_mode", type=str, default="unified", choices=["bce", "bce_dice", "unified"])
    p.add_argument("--epochs", type=int, default=None, help="Training epochs (default from config)")
    p.add_argument("--max_iterations", type=int, default=None, help="Override: max steps (epochs ignored if set)")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--save_tag", type=str, default="ascent_full")
    p.add_argument("--dilation_rates", type=str, default=None,
                   help="ASPP dilation ablation: e.g. '1' or '1,6,12,18' (default: full [1,3,6,12,18])")
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--aux_weight", type=float, default=0.25, help="Weight for auxiliary output losses (default 0.25 for small datasets)")
    p.add_argument("--light", action="store_true", help="Light model: ResNet18, 3 dilations, 128ch (faster, lighter)")
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    train_dataset = CustomDataset(config.TRAIN_FILENAME)
    train_ds = CacheDataset(train_dataset, transform=config.TRAIN_TRANSFORMS, num_workers=4, cache_rate=0.5)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    steps_per_epoch = len(train_loader)
    if args.max_iterations is not None:
        max_iter = args.max_iterations
    else:
        epochs = args.epochs if args.epochs is not None else getattr(config, "MAX_EPOCHS", 600)
        max_iter = epochs * steps_per_epoch

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    results_dir = os.path.join(config.BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    val_dataset = CustomDataset(config.VALID_FILENAME)
    val_ds = CacheDataset(val_dataset, num_workers=0, cache_rate=0.5)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    dilation_rates = None
    if args.dilation_rates:
        dilation_rates = [int(x.strip()) for x in args.dilation_rates.split(",")]

    if args.light:
        backbone_name = "resnet18"
        aspp_channels = 128
        low_level_channels = 24
        if dilation_rates is None:
            dilation_rates = [1, 6, 12]
    else:
        backbone_name = args.backbone
        aspp_channels = 256
        low_level_channels = 48

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device(f"cuda:{config.DEVICE_IDX}" if torch.cuda.is_available() else "cpu")

    model = AscentPlus(
        in_channels=3, num_classes=1, output_stride=16, pretrained=True,
        backbone_name=backbone_name, use_contrast_boost=args.use_contrast_boost, use_spam=args.use_spam,
        dilation_rates=dilation_rates, aspp_channels=aspp_channels, low_level_channels=low_level_channels,
    ).to(device)

    torch.backends.cudnn.benchmark = True
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-4},
            {"params": head_params, "lr": 1e-3, "weight_decay": 1e-4},
        ])
        max_lr = [1e-4, 1e-3]  # backbone 1e-4, head 1e-3 (10x for task layers)
    else:
        optimizer = torch.optim.SGD([
            {"params": backbone_params, "lr": 0.005, "momentum": 0.9, "weight_decay": 1e-4},
            {"params": head_params, "lr": 0.05, "momentum": 0.9, "weight_decay": 1e-4},
        ])
        max_lr = [0.005, 0.05]
    scheduler = schedulers.CosineAnnealingLR(
        optimizer, max_lr=max_lr, total_steps=max_iter,
        pct_start=0.30, div_factor=5, final_div_factor=20
    )

    train_loss_vals, val_loss_vals = [], []
    train_dice_vals, train_iou_vals = [], []
    val_dice_vals, val_iou_vals = [], []
    eval_steps = []
    global_step = config.GLOBAL_STEP
    val_loss_best = 10.0
    global_step_best = 0
    ckpt_name = f"best_metric_model_{args.save_tag}.pth"
    csv_path = os.path.join(results_dir, f"train_ascent_{args.save_tag}.csv")

    def validation(epoch_iterator_val):
        model.eval()
        total_loss, total_dice, total_iou, n = 0, 0, 0, 0
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inp = batch["image"].type(torch.FloatTensor)
                val_lbl = batch["label"].type(torch.FloatTensor)
                val_inputs, val_labels = (val_inp.to(device), val_lbl.to(device))
                dv0, dv1, dv2, dv3, dv4, dv5, dv6 = model(val_inputs)
                _, loss = muti_loss_fusion(dv0, dv1, dv2, dv3, dv4, dv5, dv6, val_labels, loss_mode=args.loss_mode, aux_weight=args.aux_weight)
                dice_pct, iou_pct = dice_iou_from_logits(dv0, val_labels)
                total_loss += loss.item()
                total_dice += dice_pct
                total_iou += iou_pct
                n += 1
        model.train()
        return total_loss / n, total_dice / n, total_iou / n

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"])

    model.train()
    epoch_loss, epoch_dice, epoch_iou, epoch_step = 0, 0, 0, 0
    pbar = tqdm(total=max_iter, dynamic_ncols=True, desc="Training ASCENT+")

    while global_step < max_iter:
        for batch in train_loader:
            if global_step >= max_iter:
                break
            epoch_step += 1
            x_img = batch["image"].type(torch.FloatTensor)
            y_img = batch["label"].type(torch.FloatTensor)
            x, y = (x_img.to(device), y_img.to(device))

            optimizer.zero_grad()
            d0, d1, d2, d3, d4, d5, d6 = model(x)
            _, loss = muti_loss_fusion(d0, d1, d2, d3, d4, d5, d6, y, loss_mode=args.loss_mode, aux_weight=args.aux_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            scheduler.step()

            with torch.no_grad():
                dice_pct, iou_pct = dice_iou_from_logits(d0, y)
            epoch_loss += loss.item()
            epoch_dice += dice_pct
            epoch_iou += iou_pct

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice_pct:.1f}")
            del d0, d1, d2, d3, d4, d5, d6, loss, x_img, y_img, x, y
            global_step += 1

            if (global_step % args.eval_every == 0 and global_step != 0) or global_step == max_iter:
                val_loss, val_dice, val_iou = validation(tqdm(val_loader, desc="Val", leave=False))
                train_avg = epoch_loss / epoch_step if epoch_step > 0 else 0
                train_dice_avg = epoch_dice / epoch_step if epoch_step > 0 else 0
                train_iou_avg = epoch_iou / epoch_step if epoch_step > 0 else 0

                eval_steps.append(global_step)
                train_loss_vals.append(train_avg)
                val_loss_vals.append(val_loss)
                train_dice_vals.append(train_dice_avg)
                train_iou_vals.append(train_iou_avg)
                val_dice_vals.append(val_dice)
                val_iou_vals.append(val_iou)

                current_epoch = global_step / steps_per_epoch
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, round(current_epoch, 2), train_avg, train_dice_avg, train_iou_avg, val_loss, val_dice, val_iou])

                tqdm.write(
                    f"Epoch {current_epoch:.1f} (step {global_step}): train_loss={train_avg:.4f} train_dice={train_dice_avg:.2f}% | "
                    f"val_loss={val_loss:.4f} val_dice={val_dice:.2f}% val_iou={val_iou:.2f}%"
                )
                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, ckpt_name))
                    tqdm.write(f"  -> Saved best to {ckpt_name}")
                epoch_loss, epoch_dice, epoch_iou, epoch_step = 0, 0, 0, 0
            if global_step >= max_iter:
                break

    pbar.close()
    epochs_vals = [s / steps_per_epoch for s in eval_steps]

    import subprocess
    ret = subprocess.run([sys.executable, "eval_single.py", "--save_tag", args.save_tag], cwd=config.BASE_DIR)
    if ret.returncode == 0:
        print("Test metrics saved.")
    else:
        print("Warning: eval_single failed (run manually after training)")


if __name__ == "__main__":
    main()
