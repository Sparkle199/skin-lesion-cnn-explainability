"""Runs SHAP against a real trained model for the first time (Stage 8 gap flagged
since 2026-07-31: src.xai.shap_explain was only ever wiring-tested against a stub
predict function, never a real trained model). For a small sample of HAM10000
validation images, produces a saved Grad-CAM overlay and a saved SHAP overlay, plus an
IoU/Dice faithfulness score for each against the ground-truth lesion segmentation mask
-- directly comparable numbers, since both use the same
src.xai.faithfulness.compute_overlap function.

Scoped to the seven-class task only: the binary task's single-sigmoid-output head
(Dense(1, ...)) doesn't cleanly match build_explainer's output_names=["benign",
"malignant"] (2 names for a 1-column predict() output) -- an untested combination,
flagged as an open item rather than risked in this first real run.

SHAP with the Image/Partition masker is comparatively slow (max_evals per image) --
keep --n-samples small.

Usage: python -m src.run_shap_explain --architecture resnet50 --n-samples 5
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src import config
from src.data import ham10000
from src.json_utils import numpy_json_default
from src.train import prepare_data
from src.xai.faithfulness import compute_overlap
from src.xai.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.xai.shap_explain import build_explainer, explain_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=list(config.IMAGE_SIZE))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--max-evals", type=int, default=500)
    parser.add_argument("--output-dir", default="results/xai_overlays")
    args = parser.parse_args()

    model_path = args.model_path or f"models/{args.architecture}_seven_class.keras"
    image_size = config.IMAGE_SIZE[args.architecture]
    classes = config.SEVEN_CLASSES

    model = tf.keras.models.load_model(model_path)
    base = model.get_layer(config.BACKBONE_LAYER_NAME[args.architecture])

    _, val_df = prepare_data("seven_class")
    sample = val_df.sample(n=args.n_samples, random_state=config.RANDOM_SEED).reset_index(drop=True)

    out_dir = Path(args.output_dir) / args.architecture
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for _, row in sample.iterrows():
        raw = tf.io.read_file(row["image_path"])
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, image_size).numpy().astype("uint8")
        images.append(img)
    images_arr = np.stack(images)

    print(f"Building SHAP explainer for {args.architecture}...")
    explainer = build_explainer(model, image_size + (3,), classes)
    print(f"Running SHAP over {len(images_arr)} images (max_evals={args.max_evals})...")
    shap_result = explain_images(explainer, images_arr, max_evals=args.max_evals, batch_size=50)
    shap_values = np.array(shap_result.values)
    print(f"SHAP values shape: {shap_values.shape}")

    results = []
    for i, row in sample.iterrows():
        image = images_arr[i]
        raw_batch = image[np.newaxis, ...].astype("float32")

        proba = model.predict(raw_batch, verbose=0)[0]
        pred_index = int(np.argmax(proba))

        heatmap = make_gradcam_heatmap(model, base, args.architecture, tf.constant(raw_batch), pred_index=pred_index)
        gradcam_overlay = overlay_heatmap(image, heatmap)

        # SHAP values for this image, predicted class: (H, W, C) -> single 2D map,
        # normalised to [0, 1] the same way Grad-CAM's heatmap is, so both are
        # comparable inputs to compute_overlap's fixed 0.5 threshold.
        shap_vals = shap_values[i, ..., pred_index]
        shap_map = np.abs(shap_vals).sum(axis=-1)
        shap_map = shap_map / (shap_map.max() + 1e-8)
        shap_overlay = overlay_heatmap(image, shap_map)

        mask = ham10000.load_segmentation_mask(row["image_id"])

        entry = {
            "image_id": row["image_id"],
            "true_class": row["dx"],
            "predicted_class": classes[pred_index],
            "gradcam_faithfulness": compute_overlap(heatmap, mask),
            "shap_faithfulness": compute_overlap(shap_map, mask),
        }
        results.append(entry)

        cv2.imwrite(str(out_dir / f"{row['image_id']}_gradcam.png"), cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{row['image_id']}_shap.png"), cv2.cvtColor(shap_overlay, cv2.COLOR_RGB2BGR))

        print(
            f"{row['image_id']}: true={row['dx']} pred={classes[pred_index]} "
            f"gradcam_iou={entry['gradcam_faithfulness']['iou']:.3f} "
            f"shap_iou={entry['shap_faithfulness']['iou']:.3f}"
        )

    summary_path = Path(f"results/shap_faithfulness_{args.architecture}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=numpy_json_default)
    print(f"Wrote {summary_path}")
    print(f"Overlay images saved to {out_dir}/")


if __name__ == "__main__":
    main()
