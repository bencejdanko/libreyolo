"""
convert_to_vela.py

Converts runs/train/yolo26_exp6/weights/last.pt
    → yolo26n_fp32.tflite              (float32, litert-torch)
    → yolo26n_int8.tflite              (static INT8, litert-torch PT2E)
    → vela_output/yolo26n_int8_vela.tflite  (Ethos-U55-256, vela CLI)

Usage:
    ~/venv/bin/python convert_to_vela.py [options]

Options:
    --pt PATH           checkpoint path  (default: runs/train/yolo26_exp6/weights/last.pt)
    --imgsz N           input resolution (default: 192)
    --calib-dir DIR     folder of .jpg/.png calibration images
                        (default: auto-discover from run tree, or random noise)
    --calib-n N         max calibration images (default: 100)
    --out-dir DIR       output directory (default: .)
    --vela-config FILE  custom vela .ini (optional; script writes built-in default)
    --skip-fp32         skip FP32 TFLite step
    --skip-int8         skip INT8 TFLite step
    --skip-vela         skip Vela compilation step
"""

import argparse
import glob
import os
import random
import subprocess
import sys

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model preparation helpers
# ---------------------------------------------------------------------------

def swap_silu_for_relu6(module: nn.Module) -> None:
    """Recursively replace SiLU with ReLU6 in-place."""
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU6(inplace=True))
        else:
            swap_silu_for_relu6(child)


def patch_for_export(net: nn.Module) -> None:
    """Patch net forward for clean trace-based export.

    At inference (export=True), YOLO26Detect.forward already runs the correct
    one-to-one branch and returns (decoded_boxes, raw_outputs).  We only need
    to unwrap that tuple and return the decoded tensor `y`.
    """
    _orig_net = net.forward

    def _patched_net(x, targets=None):
        if net.training and targets is not None:
            return _orig_net(x, targets)
        p3, p4, p5 = net.backbone(x)
        n3, n4, n5 = net.neck(p3, p4, p5)
        output = net.head([n3, n4, n5])
        if net.training:
            return output
        # At export the head returns (y, exclusive_outputs); discard the raw grid
        if isinstance(output, tuple):
            y, _ = output
        else:
            y = output
        return y  # shape: (1, 4+nc, total_anchors) — decoded NMS-free predictions

    net.forward = _patched_net


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_calib_images(calib_dir: str, imgsz: int, n: int):
    try:
        import cv2
    except ImportError:
        print("[WARN] opencv not available; skipping real calibration images.")
        return None

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))

    if not paths:
        print(f"[WARN] No images found under {calib_dir}.")
        return None

    random.seed(42)
    random.shuffle(paths)
    paths = paths[:n]

    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz))
        imgs.append(torch.from_numpy(img.astype("float32") / 255.0).permute(2, 0, 1))

    print(f"[INFO] Loaded {len(imgs)} calibration images from {calib_dir}")
    return imgs


# ---------------------------------------------------------------------------
# Export steps
# ---------------------------------------------------------------------------

def _load_model(pt_path: str, imgsz: int):
    from libreyolo import LibreYOLO26

    print(f"[INFO] Loading checkpoint: {pt_path}")
    model = LibreYOLO26(model_path=pt_path, size="n")
    swap_silu_for_relu6(model.model)
    patch_for_export(model.model)

    net = model.model
    net.eval()
    net.cpu()  # BN buffers from GPU checkpoint must match the CPU dummy tensor

    if hasattr(net, "head"):
        net.head.export = True  # triggers one-to-one branch in YOLO26Detect.forward

    dummy = torch.zeros(1, 3, imgsz, imgsz)  # CPU
    return net, dummy


def export_fp32(net, dummy, out_path: str) -> None:
    import litert_torch

    print("\n[STEP 1] FP32 TFLite …")
    edge_model = litert_torch.convert(net, (dummy,), strict_export=False)
    edge_model.export(out_path)
    print(f"[OK] {out_path}  ({os.path.getsize(out_path)/1024/1024:.2f} MB)")


def export_int8(fp32_tflite_path: str, out_path: str, calib_images, imgsz: int) -> None:
    """Quantize a FP32 TFLite to full INT8 using ai_edge_quantizer.

    Operates directly on the TFLite flatbuffer — no PT2E graph surgery needed.
    ALL supported ops (conv, relu6, reshape, transpose …) get int8 activations.
    """
    from ai_edge_quantizer import quantizer as aq
    from ai_edge_quantizer import qtyping

    print("\n[STEP 2] INT8 TFLite (ai_edge_quantizer) …")

    qt = aq.Quantizer(fp32_tflite_path)

    # Apply static INT8 to every supported op (weights + activations)
    qt.add_static_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=8,
        weight_num_bits=8,
        weight_granularity=qtyping.QuantGranularity.CHANNELWISE,
    )

    # Calibration data: dict[signature_key -> list of {input_name: np.array}]
    import numpy as np
    calib_np = [img.numpy() for img in calib_images]  # list of (3, H, W) float32

    # The TFLite model's input is (1, 3, H, W) named "args_0"
    calib_feed = [{"args_0": img[np.newaxis]} for img in calib_np]
    calib_data = {"serving_default": calib_feed}

    print(f"[INFO] Calibrating with {len(calib_feed)} images …")
    calib_result = qt.calibrate(calib_data)

    result = qt.quantize(calib_result, serialize_to_path=out_path)
    print(f"[OK] {out_path}  ({os.path.getsize(out_path)/1024/1024:.2f} MB)")


def compile_vela(int8_path: str, out_dir: str, vela_ini: str) -> str:
    print("\n[STEP 3] Vela compilation …")
    cmd = [
        "vela", int8_path,
        "--accelerator-config", "ethos-u55-256",
        "--optimise", "Size",
        "--config", vela_ini,
        "--memory-mode", "Shared_Sram",
        "--system-config", "Ethos_U55_High_End_Embedded",
        "--output-dir", out_dir,
    ]
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    stem = os.path.splitext(os.path.basename(int8_path))[0]
    vela_out = os.path.join(out_dir, f"{stem}_vela.tflite")
    if not os.path.exists(vela_out):
        raise FileNotFoundError(f"Expected Vela output not found: {vela_out}")
    print(f"[OK] {vela_out}  ({os.path.getsize(vela_out)/1024/1024:.2f} MB)")
    return vela_out


DEFAULT_VELA_INI = """\
[System_Config.Ethos_U55_High_End_Embedded]
core_clock=200e6
axi0_port=Sram
axi1_port=OffChipFlash
Sram_clock_scale=1.0
Sram_burst_length=32
Sram_read_latency=32
Sram_write_latency=32
OffChipFlash_clock_scale=0.125
OffChipFlash_burst_length=128
OffChipFlash_read_latency=64
OffChipFlash_write_latency=64

[Memory_Mode.Shared_Sram]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0
arena_cache_size=4194304
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pt", default="runs/train/yolo26_exp6/weights/last.pt")
    p.add_argument("--imgsz", type=int, default=192)
    p.add_argument("--calib-dir", default=None)
    p.add_argument("--calib-n", type=int, default=100)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--vela-config", default=None)
    p.add_argument("--skip-fp32", action="store_true")
    p.add_argument("--skip-int8", action="store_true")
    p.add_argument("--skip-vela", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Vela config
    if args.vela_config and os.path.exists(args.vela_config):
        vela_ini = args.vela_config
    else:
        vela_ini = os.path.join(args.out_dir, "default_vela.ini")
        with open(vela_ini, "w") as f:
            f.write(DEFAULT_VELA_INI)
        print(f"[INFO] Vela config → {vela_ini}")

    fp32_out = os.path.join(args.out_dir, "yolo26n_fp32.tflite")
    int8_out = os.path.join(args.out_dir, "yolo26n_int8.tflite")
    vela_dir = os.path.join(args.out_dir, "vela_output")
    os.makedirs(vela_dir, exist_ok=True)

    # Only load the PyTorch model if the FP32 TFLite step is needed
    need_pytorch = not args.skip_fp32 or (not args.skip_int8 and not os.path.exists(fp32_out))
    if need_pytorch:
        net, dummy = _load_model(args.pt, args.imgsz)
    else:
        net, dummy = None, None

    # Calibration images (needed for INT8 step)
    calib_images = None
    if not args.skip_int8:
        calib_dir = args.calib_dir
        if calib_dir is None:
            for c in ["runs/train/yolo26_exp6", "datasets", "data"]:
                if os.path.isdir(c):
                    calib_dir = c
                    break
        if calib_dir:
            calib_images = load_calib_images(calib_dir, args.imgsz, args.calib_n)
        if not calib_images:
            print("[INFO] Using 32 random-noise tensors for calibration.")
            calib_images = [torch.rand(3, args.imgsz, args.imgsz) for _ in range(32)]

    if not args.skip_fp32:
        export_fp32(net, dummy, fp32_out)

    if not args.skip_int8:
        if not os.path.exists(fp32_out):
            print(f"[ERROR] FP32 TFLite not found at {fp32_out}; run without --skip-fp32 first.")
            sys.exit(1)
        export_int8(fp32_out, int8_out, calib_images, args.imgsz)

    if not args.skip_vela:
        if not os.path.exists(int8_out):
            print(f"[ERROR] INT8 TFLite not found at {int8_out}; cannot run Vela.")
            sys.exit(1)
        compile_vela(int8_out, vela_dir, vela_ini)

    print("\n[DONE]")
    for f in [fp32_out, int8_out] + glob.glob(os.path.join(vela_dir, "*.tflite")):
        if os.path.exists(f):
            print(f"  {f}  ({os.path.getsize(f)/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
