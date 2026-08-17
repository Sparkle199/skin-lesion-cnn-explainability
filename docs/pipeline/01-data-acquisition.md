# Stage 1 — Data Acquisition

**Status:** Manual step, not yet automated.

## Purpose
Get both datasets from their archive form into the directory layout `src/config.py`
expects, so every downstream module can find them without path changes.

## Inputs
- `archive (1).zip` (3.2GB, local, gitignored) — HAM10000 release: images, metadata,
  lesion segmentation masks, and the official ISIC2018 Task 3 held-out test set.
- `ddidiversedermatologyimages.zip` (237MB, local, gitignored) — DDI: 656 images +
  `ddi_metadata.csv`.

## Process
1. Extract `archive (1).zip` such that the following paths exist (see
   `src/config.py`):
   - `data/raw/dataverse_files/HAM10000_images_combined_600x450/`
   - `data/raw/dataverse_files/HAM10000_metadata`
   - `data/raw/dataverse_files/HAM10000_segmentations_lesion_tschandl/`
   - `data/raw/dataverse_files/ISIC2018_Task3_Test_Images/`
   - `data/raw/dataverse_files/ISIC2018_Task3_Test_GroundTruth.csv`
2. Extract `ddidiversedermatologyimages.zip` such that:
   - `data/raw/ddi/` contains the 656 images
   - `data/raw/ddi/ddi_metadata.csv` exists
3. On Runpod, set the `PROJECT_DATA_DIR` environment variable to wherever the pod's
   persistent volume mounts the extracted data (default, if unset, is `data/raw`
   relative to the working directory). Extract once onto persistent storage — don't
   re-extract the 3.2GB archive on every pod restart.

## Outputs
A populated `data/raw/` tree (or whatever `PROJECT_DATA_DIR` points to) matching the
paths in `src/config.py`.

## Code references
- `src/config.py` — `DATA_ROOT`, `HAM10000_IMAGE_DIR`, `HAM10000_METADATA_CSV`,
  `HAM10000_SEGMENTATION_DIR`, `ISIC2018_TEST_IMAGE_DIR`,
  `ISIC2018_TEST_GROUNDTRUTH_CSV`, `DDI_IMAGE_DIR`, `DDI_METADATA_CSV`.

## Notes / open items
- Not yet automated (no extraction script). Worth a small `scripts/extract_data.sh` or
  equivalent if this is repeated across multiple pod sessions.
- No validation step currently confirms extraction succeeded (e.g. expected file
  counts). Consider adding a quick sanity check (image count == metadata row count,
  etc.) before kicking off a multi-hour training run on bad/partial data.
