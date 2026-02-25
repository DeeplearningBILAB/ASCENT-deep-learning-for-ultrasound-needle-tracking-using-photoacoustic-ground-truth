import os
import cv2
import torch
import numpy as np
import config
from utils.loader import CustomDataset
from monai.data import ( DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch,)
from tqdm import tqdm
from network.ascent_plus import AscentPlus
from utils.misc import overlay, save_mat


BASE_DIR = config.BASE_DIR
TEST_FILENAME = config.TEST_FILENAME
TEST_DIR = os.path.join(config.BASE_DIR, "results", "test")
BEST_MODEL = os.path.join(config.MODEL_DIR, config.MODEL_NAME)

os.makedirs(TEST_DIR, exist_ok=True)
test_dataset = CustomDataset(TEST_FILENAME)
test_ds = CacheDataset(test_dataset, num_workers=0, cache_rate=0.5)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_iterator = tqdm(test_loader, desc="Testing (X / X Steps) (loss=X.X)", dynamic_ncols=True)
model = AscentPlus(in_channels=3, num_classes=1, output_stride=16, pretrained=True).to(device)
torch.backends.cudnn.benchmark = True
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn
def clr_2_bw(image, threshold_value):
    (_, blackAndWhiteImage) = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY)
    return blackAndWhiteImage
def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    new_mask = np.zeros_like(mask, dtype=np.uint8)
    new_mask[labels == largest_label] = 1
    return new_mask

model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
model.eval()
i = 0
with torch.no_grad():
    for batch in epoch_iterator:
        i +=1
        img = batch["image"].type(torch.FloatTensor)
        label = batch["label"].type(torch.FloatTensor)
        d0, d1, d2, d3, d4, d5, d6 = model(img.cuda())
        pred = torch.sigmoid(d0[:, 0, :, :])
        pred = normPRED(pred)
        pred_numpy = pred.permute(1, 2, 0).cpu().numpy()
        pred_bin = clr_2_bw(pred_numpy, threshold_value=0.5)
        pred = keep_largest_component(pred_bin)
        img = img.squeeze(0).permute(1,2,0).cpu().numpy().squeeze()
        label = label.squeeze(0).permute(1,2,0).cpu().numpy().squeeze()

        overlay_prdctd = overlay(img, pred, color=(0, 0, 255), alpha=0.8)
        overlay_lbld = overlay(img, label)
        save_mat(file=overlay_prdctd , i = i, dir =TEST_DIR, folder_name = "overlay_predictions")
        save_mat(file=overlay_lbld, i=i, dir=TEST_DIR, folder_name="overlay_labels")
        save_mat(file=img, i=i, dir=TEST_DIR, folder_name="images")
        save_mat(file=label, i=i, dir=TEST_DIR, folder_name="labels")
        save_mat(file=pred, i=i, dir=TEST_DIR, folder_name="predictions")
    print("Testing completed")