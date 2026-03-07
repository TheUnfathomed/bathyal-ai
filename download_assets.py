"""Downloads detector assets, sample images, and BioCLIP 2 weights for Bathyal AI."""

from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).parent
BIOCLIP_MODEL_ID = "hf-hub:imageomics/bioclip-2"
SETUPTOOLS_COMMANDS = {"bdist_wheel", "develop", "dist_info", "editable_wheel", "egg_info", "sdist"}

MODELS = {
    "megalodon": {
        "repo": "FathomNet/megalodon",
        "file": "mbari-megalodon-yolov8x.pt",
    },
    "benthic-supercategory-detector": {
        "repo": "FathomNet/MBARI-benthic-supercategory-detector",
    },
    "midwater-supercategory-detector": {
        "repo": "FathomNet/MBARI-midwater-supercategory-detector",
    },
}

SAMPLE_CONCEPTS = ["Nanomia bijuga", "Sebastolobus", "Dosidicus gigas"]


def DownloadModels() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    for Name, Config in MODELS.items():
        Dest = ROOT / "models" / Name
        if Dest.exists() and any(Dest.iterdir()):
            print(f"[skip] {Name} already exists")
            continue
        print(f"[download] {Name} from {Config['repo']}")
        if "file" in Config:
            hf_hub_download(repo_id=Config["repo"], filename=Config["file"], local_dir=str(Dest))
        else:
            snapshot_download(repo_id=Config["repo"], local_dir=str(Dest))
        print(f"[done] {Name}")


def DownloadSamples() -> None:
    from fathomnet.api import images

    SamplesDir = ROOT / "samples"
    SamplesDir.mkdir(exist_ok=True)
    for Concept in SAMPLE_CONCEPTS:
        Filename = Concept.replace(" ", "_") + ".png"
        Filepath = SamplesDir / Filename
        if Filepath.exists():
            print(f"[skip] {Filename} already exists")
            continue
        Results = images.find_by_concept(Concept)
        if not Results:
            print(f"[warn] no images found for {Concept}")
            continue
        print(f"[download] {Concept} -> {Filename}")
        urllib.request.urlretrieve(Results[0].url, str(Filepath))
        print(f"[done] {Filename}")


def DownloadBioClip2() -> None:
    try:
        import open_clip
        import torch
    except ImportError:
        print("[skip] open_clip_torch is not installed yet; install project dependencies first")
        return

    print(f"[download] BioCLIP 2 from {BIOCLIP_MODEL_ID}")
    try:
        model, _ = open_clip.create_model_from_pretrained(BIOCLIP_MODEL_ID)
    except Exception:
        model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print("[done] BioCLIP 2")


def main() -> None:
    DownloadModels()
    DownloadSamples()
    DownloadBioClip2()
    print("\nSetup complete. Run inference with:")
    print("  bathyal-ai --source samples/wild --support-dir samples --similarity-threshold 0.7 --margin-threshold 0.03")


if __name__ == "__main__" and not (SETUPTOOLS_COMMANDS & set(sys.argv[1:])):
    main()
