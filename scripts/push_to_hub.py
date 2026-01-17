import argparse
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file

def create_config(num_chars=0):
    return {
        "height": 32,
        "hidden_size": 256,
        "vocab_size": num_chars,
        "model_type": "kiri-ocr",
        "architectures": ["LightweightOCR"],
        "framework": "pytorch"
    }

def create_model_card(repo_id, model_name="Kiri OCR Model", benchmark_images=None):
    content = f"""---
language:
- en
- kh
tags:
- ocr
- pytorch
- handwritten
license: apache-2.0
datasets:
- mrrtmob/km_en_image_line
---

# {model_name}

**Kiri OCR** is a lightweight, OCR library for **English and Khmer** documents. It provides document-level text detection, recognition, and rendering capabilities in a compact package (~13MB model).

## ✨ Key Features

- **Lightweight**: Only ~13MB model size (Lite version).
- **Bi-lingual**: Native support for English and Khmer (and mixed).
- **Document Processing**: Automatic text line and word detection.
- **Robust Detection**: Works on both light and dark backgrounds (Dark Mode support).
- **Visualizations**: Generate annotated images and HTML reports.

## 📊 Dataset

The model is trained on the [mrrtmob/km_en_image_line](https://huggingface.co/datasets/mrrtmob/km_en_image_line) dataset, which contains **5 million** synthetic images of Khmer and English text lines.

## 💻 Usage

### Installation

```bash
pip install kiri-ocr
```

### Python API

```python
from kiri_ocr import OCR

# Initialize (loads from Hugging Face automatically)
ocr = OCR()

# Extract text
text, results = ocr.extract_text('document.jpg')
print(text)
```

### CLI Tool

```bash
kiri-ocr predict path/to/document.jpg --output results/
```

## Model Details
- **Architecture**: CRNN (CNN + LSTM + CTC)
- **Framework**: PyTorch
- **Input Size**: Height 32px (width variable)
"""
    if benchmark_images:
        content += "\n## 📈 Benchmarks\n\n"
        content += "Results on synthetic test images (10 popular fonts):\n\n"
        for img_name in benchmark_images:
            content += f"![{img_name}]({img_name})\n\n"
            
    return content

def main():
    parser = argparse.ArgumentParser(description="Push Kiri OCR model to Hugging Face Hub")
    parser.add_argument("repo_id", nargs='?', default="mrrtmob/kiri-ocr", help="Hugging Face repo ID (default: mrrtmob/kiri-ocr)")
    parser.add_argument("--model", default="kiri_ocr/models/model.kiri", help="Path to model file")
    parser.add_argument("--token", help="Hugging Face API token (optional if logged in)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Preparing to push {model_path} to {args.repo_id}...")
    
    api = HfApi(token=args.token)
    
    # Create repo if not exists
    try:
        create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True, token=args.token)
        print(f"✅ Repository {args.repo_id} ready")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Upload model
    print("⬆️ Uploading model file...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.kiri",
        repo_id=args.repo_id,
        token=args.token
    )

    # Upload config.json (for download tracking and metadata)
    print("⚙️ Uploading config.json...")
    config = create_config() # We could load actual charset size if we loaded the model, but generic is fine for now
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    api.upload_file(
        path_or_fileobj="config.json",
        path_in_repo="config.json",
        repo_id=args.repo_id,
        token=args.token
    )
    os.remove("config.json")
    
    # Upload Benchmarks
    benchmark_dir = Path("benchmark")
    benchmark_images = []
    for img_name in ["benchmark_table.png", "benchmark_graph.png"]:
        img_path = benchmark_dir / img_name
        if img_path.exists():
            print(f"📊 Uploading benchmark image: {img_name}")
            api.upload_file(
                path_or_fileobj=img_path,
                path_in_repo=img_name,
                repo_id=args.repo_id,
                token=args.token
            )
            benchmark_images.append(img_name)

    # Upload Model Card
    print("📝 Creating and uploading model card...")
    readme_content = create_model_card(args.repo_id, benchmark_images=benchmark_images)
    with open("README_temp.md", "w") as f:
        f.write(readme_content)
        
    api.upload_file(
        path_or_fileobj="README_temp.md",
        path_in_repo="README.md",
        repo_id=args.repo_id,
        token=args.token
    )
    os.remove("README_temp.md")
    
    print(f"\n✅ Successfully pushed to https://huggingface.co/{args.repo_id}")
    print("\nYou can now load this model in Kiri OCR using:")
    print(f"ocr = OCR(model_path='{args.repo_id}')")

if __name__ == "__main__":
    main()
