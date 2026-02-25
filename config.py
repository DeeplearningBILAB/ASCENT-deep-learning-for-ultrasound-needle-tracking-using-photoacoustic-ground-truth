from monai.transforms import (
    Compose,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    RandScaleIntensityd,
)

BASE_DIR = r""

TRAIN_FILENAME = r""
TEST_FILENAME = r""
VALID_FILENAME = r""
MODEL_NAME = r"best_metric_model.pth"
MODEL_DIR = r""

DEVICE_IDX = 0
MAX_EPOCHS = 600
EVALUATION_NUM = 10

GLOBAL_STEP = 0

# 'resnet50', 'resnet101', 'mobilenetv3'
BACKBONE_NAME = 'resnet50'

BACKBONE_MODEL_NAMES = {
    'resnet50': 'best_metric_model_resnet50.pth',
    'resnet101': 'best_metric_model_resnet101.pth',
    'mobilenetv3': 'best_metric_model_mobilenetv3.pth'
}

TRAIN_TRANSFORMS = Compose([
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.15),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.15),
    RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    RandScaleIntensityd(keys=["image"], factors=(-0.15, 0.15), prob=0.40),
])