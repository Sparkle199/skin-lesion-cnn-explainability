"""Paths, class lists, and label mappings shared across the data pipeline.

Per spec/specification.md: HAM10000 is the primary (seven-class) corpus; DDI is used
only for the secondary binary malignant/benign task, not merged into the seven classes.
"""

import os
from pathlib import Path

# Root under which both datasets live, however they were made available:
# extracted locally, or mounted read-only under Kaggle's /kaggle/input/.
# Override with the PROJECT_DATA_DIR environment variable.
DATA_ROOT = Path(os.environ.get("PROJECT_DATA_DIR", "data/raw"))

# --- HAM10000 (from archive (1).zip -> dataverse_files/) ---
HAM10000_IMAGE_DIR = DATA_ROOT / "dataverse_files" / "HAM10000_images_combined_600x450"
HAM10000_METADATA_CSV = DATA_ROOT / "dataverse_files" / "HAM10000_metadata"
HAM10000_SEGMENTATION_DIR = DATA_ROOT / "dataverse_files" / "HAM10000_segmentations_lesion_tschandl"
ISIC2018_TEST_IMAGE_DIR = DATA_ROOT / "dataverse_files" / "ISIC2018_Task3_Test_Images"
ISIC2018_TEST_GROUNDTRUTH_CSV = DATA_ROOT / "dataverse_files" / "ISIC2018_Task3_Test_GroundTruth.csv"

# --- DDI (from ddidiversedermatologyimages.zip) ---
DDI_IMAGE_DIR = DATA_ROOT / "ddi"
DDI_METADATA_CSV = DATA_ROOT / "ddi" / "ddi_metadata.csv"

# Primary task: HAM10000's native seven diagnostic classes.
SEVEN_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Secondary task: binary relabelling of HAM10000's seven classes.
# Confirm this grouping with the supervisor before training (see agent journal,
# 2026-07-31 entry) -- akiec is a borderline/precancerous case some papers group
# differently.
MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}
BENIGN_CLASSES = {"bkl", "df", "nv", "vasc"}

# DDI's `skin_tone` column uses three grouped Fitzpatrick bands, not the full I-VI scale.
DDI_SKIN_TONE_LABELS = {
    12: "FST_I_II",
    34: "FST_III_IV",
    56: "FST_V_VI",
}

# Per-architecture input size (EfficientNetB4's canonical input is 380x380; the other
# two are standardly used at 224x224).
IMAGE_SIZE = {
    "resnet50": (224, 224),
    "efficientnetb4": (380, 380),
    "vgg16": (224, 224),
}

# Last convolutional/activation layer in each backbone, used as the Grad-CAM target.
LAST_CONV_LAYER = {
    "resnet50": "conv5_block3_out",
    "efficientnetb4": "top_activation",
    "vgg16": "block5_conv3",
}

# keras.applications' default model name for each backbone -- used to re-fetch the
# nested backbone submodel (as a layer) from a reloaded saved model, e.g. in
# src.evaluate_run, since src.models.build.build_model does not itself persist which
# layer is the backbone. NOT yet verified against a real saved-and-reloaded model
# (requires TensorFlow, unavailable in this environment) -- confirm this holds before
# relying on it (see agent journal).
BACKBONE_LAYER_NAME = {
    "resnet50": "resnet50",
    "efficientnetb4": "efficientnetb4",
    "vgg16": "vgg16",
}

RANDOM_SEED = 42
