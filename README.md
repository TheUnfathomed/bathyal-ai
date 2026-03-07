# Bathyal AI

Bathyal AI uses a production-oriented classification flow:

1. Megalodon detects organisms.
2. A frozen BioCLIP 2 encoder embeds each detection crop.
3. A trained supervised classifier head predicts species.
4. Open-set gating accepts the species only if calibrated probability, probability margin, and class-centroid similarity all clear threshold.
5. Anything else becomes `unknown` and can fall through to your expensive secondary system.

Inference loads a trained classifier bundle. It does not depend on a live support-image folder.

## Install

```bash
uv sync
uv run python download_assets.py
```

## Dataset layout

Use grouped directories for training and validation.

```text
datasets/
  train/
    Species_One/
      crop_001.jpg
      crop_002.jpg
  val/
    Species_One/
      crop_101.jpg
  unknown_val/
    habitat_only_001.jpg
    bad_detector_crop_002.jpg
```

Guidelines:

- `train/` and `val/` should contain only species the cheap classifier is allowed to predict.
- `unknown_val/` should contain things you want rejected: bad crops, habitat-only images, partial animals, non-target taxa, and out-of-taxonomy species.
- Split by dive, expedition, or deployment when possible. Avoid near-duplicate frames across train and validation.

## Ingest data

`bathyal-ai-ingest` is the data-ingest side. It means turning raw source imagery plus annotations into classifier-ready crops, deterministic train/val splits, and a manifest with provenance.

Current built-in source:

- `fathomnet`: downloads full source images, crops verified bounding boxes, writes `train/` and `val/`, and records metadata in `manifests/fathomnet.jsonl`

Example:

```bash
uv run bathyal-ai-ingest fathomnet \
  --concept Sebastes,Actiniaria \
  --limit-per-concept 100 \
  --val-fraction 0.2 \
  --output-dir datasets/fathomnet_seed
```

Useful flags:

- `--dataset-role unknown` writes flat crops to `unknown_val/`
- `--padding-fraction 0.1` adds context around the box
- `--include-descendants --taxa-provider <provider>` expands a taxon query through a FathomNet taxonomy provider
- `--review-states VERIFIED,UNVERIFIED` controls which annotations are accepted

Recommended official data sources:

- [FathomNet](https://www.fathomnet.org/)
- [FathomNet download guide](https://www.fathomnet.org/post/how-to-download-images-and-bounding-boxes)
- [MBARI Deep-Sea Guide image search](https://dsg.mbari.org/dsg/imagesearch)
- [MBARI VARS](https://www.mbari.org/technology/video-annotation-and-reference-system-vars/)
- [NOAA Ocean Exploration](https://osrefresh.oceanexplorer.noaa.gov/video_playlist.html)
- [iNaturalist data export help](https://help.inaturalist.org/en/support/solutions/articles/151000170342-how-can-i-download-data-from-inaturalist-)
- [WoRMS](https://www.marinespecies.org/about.php) for taxonomy normalization

## Train the classifier bundle

```bash
uv run bathyal-ai-train \
  --train-dir datasets/train \
  --val-dir datasets/val \
  --unknown-dir datasets/unknown_val \
  --output-dir artifacts/species_classifier
```

This produces a bundle directory containing:

- `metadata.json`
- `head.pt`
- `centroids.npy`
- `training_report.json`

Training keeps BioCLIP 2 frozen and learns a supervised classifier head on top of embeddings. It also calibrates a temperature scalar and selects open-set thresholds from validation data.

## Evaluate before shipping

```bash
uv run bathyal-ai-eval \
  --source datasets/val \
  --unknown-dir datasets/unknown_val \
  --classifier-bundle artifacts/species_classifier
```

This reports:

- `top1` accuracy on known classes
- accepted-label precision
- known coverage
- known fallback rate
- unknown accept rate

Those are the metrics that matter if your expensive fallback only runs on `unknown`.

## Run detection + classification

```bash
uv run bathyal-ai \
  --source path/to/images \
  --classifier-bundle artifacts/species_classifier
```

Useful overrides:

- `--probability-threshold 0.85`
- `--margin-threshold 0.08`
- `--centroid-threshold 0.55`
- `--save-crops`

If you do not pass overrides, inference uses the thresholds stored in the bundle.

## Outputs

For an input image named `fishes_underwater.jpg`, the pipeline writes files like these inside the Megalodon run directory:

- `fishes_underwater.jpg` from Megalodon itself
- `fishes_underwater.species_classifier.jpg` with the accepted species name drawn over each bounding box
- `fishes_underwater.species_classifier.json` with detector scores, classifier probabilities, margins, centroid similarities, candidate rankings, and accept/reject reasons
