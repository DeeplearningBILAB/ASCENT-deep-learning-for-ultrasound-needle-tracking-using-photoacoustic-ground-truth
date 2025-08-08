import os
import numpy as np
import torch
import config
from tqdm import tqdm
from sklearn.model_selection import KFold
from model.network import AscentPlus
from matplotlib import pyplot as plt
from utils.loader import CustomDataset
from utils.loss import muti_loss_fusion
from monai.data import (DataLoader, CacheDataset)
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device(f"cuda:{config.DEVICE_IDX}" if torch.cuda.is_available() else "cpu")

K_FOLDS = 10
KFOLD_MODEL_DIR = os.path.join(config.MODEL_DIR, "kfold_models")
os.makedirs(KFOLD_MODEL_DIR, exist_ok=True)


class KFoldDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = torch.from_numpy(image).float().permute(2, 0, 1)
        label = torch.from_numpy(label).long().permute(2, 0, 1)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_data():
    train_data = np.load(config.TRAIN_FILENAME)
    train_images = train_data['arr_0']
    train_labels = train_data['arr_1']

    val_data = np.load(config.VALID_FILENAME)
    val_images = val_data['arr_0']
    val_labels = val_data['arr_1']

    test_data = np.load(config.TEST_FILENAME)
    test_images = test_data['arr_0']
    test_labels = test_data['arr_1']

    all_images = np.concatenate((train_images, val_images, test_images), axis=0)
    all_labels = np.concatenate((train_labels, val_labels, test_labels), axis=0)

    return all_images, all_labels


def validation(model, val_loader):
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
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

    return average_loss, average_dice


def train_model(model, train_loader, val_loader, fold):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    epoch_loss_values = []
    metric_values = []
    dice_val_best = float('inf')
    global_step_best = 0
    global_step = 0

    model_file = os.path.join(KFOLD_MODEL_DIR, f"fold_{fold}_model.pth")

    model.train()
    while global_step < config.MAX_ITERATIONS:
        epoch_loss = 0
        epoch_dice = 0
        epoch_step = 0

        epoch_iterator = tqdm(train_loader,
                              desc=f"Fold {fold} - Training Iteration {global_step}/{config.MAX_ITERATIONS}",
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
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                pred = (d0 > 0.5).float()
                intersection = (pred * y).sum()
                dice = (2. * intersection) / (pred.sum() + y.sum() + 1e-5)
                dice_percent = dice.item() * 100
                epoch_dice += dice_percent

            epoch_iterator.set_description(
                f"Fold {fold} - Training ({global_step}/{config.MAX_ITERATIONS}) (loss={loss.item():.5f}, acc={dice_percent:.2f}%)"
            )

            del d0, d1, d2, d3, d4, d5, d6, loss2, loss, pred, x_img, y_img, x, y

            global_step += 1

            if (global_step % config.EVALUATION_NUM == 0 and global_step != 0) or global_step == config.MAX_ITERATIONS:
                model.eval()
                epoch_iterator_val = tqdm(val_loader, desc=f"Fold {fold} - Validation", dynamic_ncols=True)
                dice_val, dice_acc = validation(model, epoch_iterator_val)

                current_epoch_avg_loss = epoch_loss / epoch_step if epoch_step > 0 else 0
                current_epoch_avg_dice = epoch_dice / epoch_step if epoch_step > 0 else 0
                epoch_loss_values.append(current_epoch_avg_loss)
                metric_values.append(dice_val)

                print(f"\nFold {fold} - Iteration {global_step}: Training Avg Loss: {current_epoch_avg_loss:.4f}, "
                      f"Training Avg Acc: {current_epoch_avg_dice:.2f}%")
                print(f"Fold {fold} - Iteration {global_step}: Validation Loss: {dice_val:.4f}, "
                      f"Validation Acc: {dice_acc:.2f}%")

                if dice_val < dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), model_file)
                    print(f"Model Saved! Fold {fold} - Iteration: {global_step}, "
                          f"Best Val Loss: {dice_val_best:.4f}, Val Acc: {dice_acc:.2f}%")
                else:
                    print(
                        f"Model Not Saved. Fold {fold} - Best Val Loss: {dice_val_best:.4f} at iter {global_step_best}, "
                        f"Current Val Loss: {dice_val:.4f}, Val Acc: {dice_acc:.2f}%")

                model.train()

                epoch_loss = 0
                epoch_dice = 0
                epoch_step = 0

            if global_step >= config.MAX_ITERATIONS:
                break

        if global_step >= config.MAX_ITERATIONS:
            break

    return dice_val_best, global_step_best


def run_kfold_cv():
    """Run k-fold cross-validation"""
    all_images, all_labels = load_data()

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(all_images)))):
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold + 1}/{K_FOLDS}")
        print(f"{'=' * 50}")

        train_images = all_images[train_idx]
        train_labels = all_labels[train_idx]
        val_images = all_images[val_idx]
        val_labels = all_labels[val_idx]

        print(f"Training set: {train_images.shape[0]} samples ({train_idx.shape[0] / len(all_images) * 100:.1f}%)")
        print(f"Validation set: {val_images.shape[0]} samples ({val_idx.shape[0] / len(all_images) * 100:.1f}%)")

        train_dataset = KFoldDataset(train_images, train_labels, transform=config.TRAIN_TRANSFORMS)
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

        val_dataset = KFoldDataset(val_images, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        model = AscentPlus(in_channels=3, num_classes=1, output_stride=16, pretrained=True).to(device)

        best_val_loss, best_iteration = train_model(model, train_loader, val_loader, fold + 1)

        model_file = os.path.join(KFOLD_MODEL_DIR, f"fold_{fold + 1}_model.pth")
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file))
            epoch_iterator_val = tqdm(val_loader, desc=f"Final Evaluation Fold {fold + 1}", dynamic_ncols=True)
            final_val_loss, final_val_acc = validation(model, epoch_iterator_val)
        else:
            final_val_loss = best_val_loss
            final_val_acc = 0

        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'final_val_acc': final_val_acc,
            'best_iteration': best_iteration
        })

        print(f"\nFold {fold + 1} completed. Best validation loss: {best_val_loss:.4f} at iteration {best_iteration}")
        print(f"Final validation loss: {final_val_loss:.4f}, accuracy: {final_val_acc:.2f}%")

    avg_val_loss = sum(result['final_val_loss'] for result in fold_results) / K_FOLDS
    avg_val_acc = sum(result['final_val_acc'] for result in fold_results) / K_FOLDS

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'=' * 50}")
    print(f"K-FOLD CROSS-VALIDATION RESULTS (K={K_FOLDS})")
    print(f"{'=' * 50}")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.2f}%")
    print(f"Total time: {total_time / 3600:.2f} hours")

    print("\nResults for each fold:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Loss = {result['final_val_loss']:.4f}, Acc = {result['final_val_acc']:.2f}%")

    plt.figure("K-Fold CV Results", (12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"{K_FOLDS}-Fold Cross-Validation Loss")
    fold_nums = [result['fold'] for result in fold_results]
    val_losses = [result['final_val_loss'] for result in fold_results]
    plt.bar(fold_nums, val_losses)
    plt.axhline(y=avg_val_loss, color='r', linestyle='-', label=f'Average: {avg_val_loss:.4f}')
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(f"{K_FOLDS}-Fold Cross-Validation Accuracy")
    val_accs = [result['final_val_acc'] for result in fold_results]
    plt.bar(fold_nums, val_accs)
    plt.axhline(y=avg_val_acc, color='r', linestyle='-', label=f'Average: {avg_val_acc:.2f}%')
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(KFOLD_MODEL_DIR, "kfold_results.png"))
    plt.show()

    return fold_results


def create_ensemble_model(fold_results, all_images, all_labels):
    best_fold = min(fold_results, key=lambda x: x['final_val_loss'])
    print(
        f"Best individual fold: Fold {best_fold['fold']} with loss {best_fold['final_val_loss']:.4f} and accuracy {best_fold['final_val_acc']:.2f}%")

    eval_dataset = KFoldDataset(all_images, all_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    ensemble_models = []
    for fold in range(1, K_FOLDS + 1):
        model_path = os.path.join(KFOLD_MODEL_DIR, f"fold_{fold}_model.pth")
        if os.path.exists(model_path):
            model = PADeepLabv3Plus(in_channels=3, num_classes=1, output_stride=16, pretrained=False).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            ensemble_models.append(model)

    total_loss = 0
    total_dice = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating ensemble model", dynamic_ncols=True):
            inp = batch["image"].type(torch.FloatTensor)
            lbl = batch["label"].type(torch.FloatTensor)
            inputs, labels = (inp.to(device), lbl.to(device))

            all_preds = []
            all_losses = []

            for model in ensemble_models:
                d0, d1, d2, d3, d4, d5, d6 = model(inputs)
                loss2, loss = muti_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
                pred = (d0 > 0.5).float()
                all_preds.append(pred)
                all_losses.append(loss.item())

            ensemble_pred = torch.mean(torch.stack(all_preds), dim=0)
            ensemble_loss = np.mean(all_losses)

            binary_pred = (ensemble_pred > 0.5).float()
            intersection = (binary_pred * labels).sum()
            dice = (2. * intersection) / (binary_pred.sum() + labels.sum() + 1e-5)
            dice_percent = dice.item() * 100

            total_loss += ensemble_loss
            total_dice += dice_percent
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    print(f"\nEnsemble Model Performance:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Accuracy: {avg_dice:.2f}%")

    best_model_path = os.path.join(KFOLD_MODEL_DIR, f"fold_{best_fold['fold']}_model.pth")
    final_model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)

    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, final_model_path)
        print(f"\nCopied best model (Fold {best_fold['fold']}) to {final_model_path}")

    ensemble_model = PADeepLabv3Plus(in_channels=3, num_classes=1, output_stride=16, pretrained=False).to(device)

    state_dicts = [model.state_dict() for model in ensemble_models]

    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

    # Load averaged weights into ensemble model
    ensemble_model.load_state_dict(avg_state_dict)

    # Save ensemble model
    ensemble_model_path = os.path.join(KFOLD_MODEL_DIR, "ensemble_model.pth")
    torch.save(ensemble_model.state_dict(), ensemble_model_path)
    print(f"Saved ensemble model to {ensemble_model_path}")

    return {
        'best_fold': best_fold['fold'],
        'best_fold_loss': best_fold['final_val_loss'],
        'best_fold_acc': best_fold['final_val_acc'],
        'ensemble_loss': avg_loss,
        'ensemble_acc': avg_dice
    }


if __name__ == "__main__":
    # Load all data
    all_images, all_labels = load_data()

    # Run 10-fold cross-validation
    fold_results = run_kfold_cv()

    ensemble_results = create_ensemble_model(fold_results, all_images, all_labels)

    print("\nK-Fold Cross-Validation and Ensemble Model Creation Complete!")
