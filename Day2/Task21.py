import argparse
import time
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import json
import os

IMAGENET_CLASS_INDEX_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def download_imagenet_labels(cache_dir: Path) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels_path = cache_dir / "imagenet_classes.txt"
    if not labels_path.exists():
        print(f"Downloading ImageNet labels -> {labels_path}")
        urllib.request.urlretrieve(IMAGENET_CLASS_INDEX_URL, labels_path)
    return labels_path.read_text(encoding="utf-8").splitlines()


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def preprocess_image(img: Image.Image) -> torch.Tensor:
    # GoogLeNet expects 224x224 and ImageNet normalization
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(img).unsqueeze(0)  # (1,3,224,224)
    return x


@torch.inference_mode()
def predict(model, x: torch.Tensor, labels: list[str], topk: int = 5):
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    top_probs, top_ids = probs.topk(topk, dim=1)
    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_ids = top_ids.squeeze(0).cpu().tolist()

    results = [(labels[i], float(p)) for i, p in zip(top_ids, top_probs)]
    return results


@torch.inference_mode()
def benchmark(model, device: str, iters: int = 200, warmup: int = 50, size: int = 224):
    # Random input like your earlier tasks (good for latency baseline)
    x = torch.randn(1, 3, size, size, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000 / iters
    fps = 1000 / avg_ms
    return avg_ms, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="", help="Path to a local image (jpg/png).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Using device:", device)

    # Load pretrained GoogLeNet (ImageNet)
    weights = models.GoogLeNet_Weights.DEFAULT
    model = models.googlenet(weights=weights, aux_logits=True).to(device)
    model.eval()

    # Labels
    labels = download_imagenet_labels(Path.home() / ".cache" / "imagenet_labels")

    # If user provided an image, run real prediction
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = load_image(img_path)
        x = preprocess_image(img).to(device)

        preds = predict(model, x, labels, topk=5)
        print(f"\nTop-5 Predictions for {img_path.name}:")
        for name, p in preds:
            print(f"  {name:30s}  {p:.4f}")
    else:
        # Test on all images in testI folder
        test_folder = Path(__file__).parent.parent / "testI"
        output_folder = Path(__file__).parent / "classification_results"
        output_folder.mkdir(exist_ok=True)
        
        all_results = []
        
        if test_folder.exists():
            test_images = sorted(test_folder.glob("*.png"))
            if test_images:
                print(f"\nTesting on {len(test_images)} images from testI folder:\n")
                
                for img_path in test_images:
                    print(f"{'='*60}")
                    print(f"Image: {img_path.name}")
                    print(f"{'='*60}")
                    
                    img = load_image(img_path)
                    x = preprocess_image(img).to(device)
                    
                    # Measure inference time
                    start = time.time()
                    preds = predict(model, x, labels, topk=5)
                    inference_time = (time.time() - start) * 1000
                    
                    print(f"Inference Time: {inference_time:.2f} ms")
                    print("\nTop-5 Predictions:")
                    for name, p in preds:
                        print(f"  {name:30s}  {p:.4f}")
                    print()
                    
                    # Save annotated image
                    img_annotated = img.copy()
                    draw = ImageDraw.Draw(img_annotated)
                    
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
                    except:
                        font = ImageFont.load_default()
                        font_small = ImageFont.load_default()
                    
                    # Draw background rectangle for text
                    text_height = 35 + len(preds) * 25
                    draw.rectangle([(0, 0), (img.width, text_height)], fill=(0, 0, 0, 200))
                    
                    # Draw predictions
                    draw.text((10, 5), f"Top-1: {preds[0][0]}", fill=(0, 255, 0), font=font)
                    y_offset = 40
                    for i, (name, prob) in enumerate(preds[1:3], 2):  # Show top 3
                        draw.text((10, y_offset), f"Top-{i}: {name} ({prob:.2%})", fill=(255, 255, 255), font=font_small)
                        y_offset += 25
                    
                    # Save annotated image
                    output_img_path = output_folder / f"{img_path.stem}_classified.png"
                    img_annotated.save(output_img_path)
                    
                    # Store results
                    all_results.append({
                        'image': img_path.name,
                        'inference_time_ms': round(inference_time, 2),
                        'top_5_predictions': [{'class': name, 'confidence': round(p, 4)} for name, p in preds]
                    })
                    
                    print(f"✓ Saved: {output_img_path.name}")
                
                # Save JSON results
                json_path = output_folder / "predictions.json"
                with open(json_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"\n✓ Saved predictions to: {json_path}")
                
                # Save text report
                report_path = output_folder / "report.txt"
                with open(report_path, 'w') as f:
                    f.write("="*60 + "\n")
                    f.write("GoogLeNet Classification Results\n")
                    f.write("="*60 + "\n\n")
                    for result in all_results:
                        f.write(f"Image: {result['image']}\n")
                        f.write(f"Inference Time: {result['inference_time_ms']} ms\n")
                        f.write(f"Top-1 Prediction: {result['top_5_predictions'][0]['class']} ({result['top_5_predictions'][0]['confidence']:.2%})\n")
                        f.write("\nTop-5 Predictions:\n")
                        for i, pred in enumerate(result['top_5_predictions'], 1):
                            f.write(f"  {i}. {pred['class']:30s} {pred['confidence']:.4f}\n")
                        f.write("\n" + "-"*60 + "\n\n")
                print(f"✓ Saved report to: {report_path}\n")

    # Benchmark speed
    avg_ms, fps = benchmark(model, device=device, iters=args.iters, warmup=args.warmup, size=224)
    print(f"\nBenchmark (GoogLeNet, 224x224): {avg_ms:.2f} ms  |  {fps:.2f} FPS")


if __name__ == "__main__":
    main()