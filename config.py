from monai.transforms import (Compose, RandFlipd,  RandShiftIntensityd)


BASE_DIR = r"<ADD_YOUR_BASE_DIR>"

TRAIN_FILENAME = r"<ADD_YOUR_TRAIN_FILENAME>"
TEST_FILENAME = r"<ADD_YOUR_TEST_FILENAME>"
VALID_FILENAME = r"<ADD_YOUR_VALID_FILENAME>"
MODEL_NAME = r"<ADD_YOUR_MODEL_NAME>"
MODEL_DIR = r"<ADD_YOUR_MODEL_DIR>"

DEVICE_IDX = 0
MAX_ITERATIONS = 12000
EVALUATION_NUM = 100

GLOBAL_STEP = 0

# 'resnet50', 'resnet101', 'mobilenetv3'
BACKBONE_NAME = 'resnet50'

BACKBONE_MODEL_NAMES = {
    'resnet50': 'best_metric_model_resnet50.pth',
    'resnet101': 'best_metric_model_resnet101.pth',
    'mobilenetv3': 'best_metric_model_mobilenetv3.pth'
}

TRAIN_TRANSFORMS = Compose([ RandFlipd(keys=["image", "label"],spatial_axis=[0], prob=0.10,),RandFlipd(keys=["image", "label"],spatial_axis=[1], prob=0.10,),RandShiftIntensityd(
    keys=["image"], offsets=0.10, prob=0.50,)])
