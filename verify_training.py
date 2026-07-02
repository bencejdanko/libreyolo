import logging
import torch
from libreyolo import LibreYOLO26

# Configure logging to show info-level training outputs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("libreyolo").setLevel(logging.INFO)

print("Initializing LibreYOLO26-s model from scratch (random weights)...")
model = LibreYOLO26(model_path=None, size="s", nb_classes=80)

print("\nStarting training for 1 epoch...")
try:
    results = model.train(
        data="local_coco.yaml",  # Configured to use your local COCO-2017 dataset
        epochs=1,
        batch=128,
        imgsz=192,          # Reduced resolution to 192x192
        device="0" if torch.cuda.is_available() else "cpu",
        workers=8,
    )
    print("\nVerification training completed successfully!")
    print(f"Results dict: {results}")
    if results and "best_checkpoint" in results and results["best_checkpoint"]:
        import shutil
        dest = "yolo26_coco_1epoch.pt"
        shutil.copy(results["best_checkpoint"], dest)
        print(f"Copied best checkpoint to: {dest}")
except Exception as e:
    print(f"\nTraining verification failed with error:\n{e}")
    raise e
