# Kiri OCR 📄

[![PyPI version](https://badge.fury.io/py/kiri-ocr.svg)](https://badge.fury.io/py/kiri-ocr)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/kiri-ocr.svg)](https://pypi.org/project/kiri-ocr/)
[![Downloads](https://static.pepy.tech/badge/kiri-ocr)](https://pepy.tech/project/kiri-ocr)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/mrrtmob/kiri-ocr)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mrrtmob/kiri-ocr)

**Kiri OCR** is a lightweight, OCR library for **English and Khmer** documents. It provides document-level text detection, recognition, and rendering capabilities in a compact package.

[**🚀 Try the Live Demo**](https://huggingface.co/spaces/mrrtmob/kiri-ocr)

![Kiri OCR](https://raw.githubusercontent.com/mrrtmob/kiri-ocr/main/assets/image.png)

## ✨ Key Features

- **Lightweight**: Compact model optimized for speed and efficiency.
- **Bi-lingual**: Native support for English and Khmer (and mixed).
- **Document Processing**: Automatic text line and word detection.
- **Easy to Use**: Simple Python API.

## 📊 Dataset

The model is trained on the [mrrtmob/khmer_english_ocr_image_line](https://huggingface.co/datasets/mrrtmob/khmer_english_ocr_image_line) dataset, which contains **12 million** synthetic images of Khmer and English text lines.

## 📈 Benchmark

Results on synthetic test images (10 popular fonts):

![Benchmark Graph](https://raw.githubusercontent.com/mrrtmob/kiri-ocr/main/benchmark/benchmark_graph.png)

![Benchmark Table](https://raw.githubusercontent.com/mrrtmob/kiri-ocr/main/benchmark/benchmark_table.png)

## 📦 Installation

Install easily via pip:

```bash
pip install kiri-ocr
```

Or install from source:

```bash
git clone https://github.com/mrrtmob/kiri-ocr.git
cd kiri-ocr
pip install .
```

## 💻 Usage

### CLI Tool (Inference)

Run OCR on an image and save results:

```bash
kiri-ocr predict path/to/document.jpg --output results/
```

*(Or simply `kiri-ocr path/to/document.jpg`)*

### Python API

```python
from kiri_ocr import OCR

# Initialize (loads from Hugging Face automatically)
ocr = OCR()

# Extract text
text, results = ocr.extract_text('document.jpg')
print(text)
```

## 🎓 Training a New Model

Follow this guide to train a custom model from scratch.

### Step 1: Generate Training Data

Create synthetic training images from a text file.

1. **Prepare text file**: Create `data/textlines.txt` with your training text (one sentence per line).
2. **Generate dataset**:

   ```bash
   kiri-ocr generate \
       --train-file data/textlines.txt \
       --output data \
       --fonts-dir fonts \
       --augment 1 \
       --random-augment
   ```

   * `--fonts-dir`: Directory containing `.ttf` files (Khmer/English fonts).
   * `--augment`: How many variations to generate per line (e.g., 2).
   * `--random-augment`: Apply random noise/rotation even if `augment` is 1.

### Custom Dataset Structure

If you have your own data (not generated), organize it as follows:

```
data/
  ├── train/
  │   ├── labels.txt       # Tab-separated: filename <tab> text
  │   └── images/          # Image files
  │       ├── img_001.png
  │       ├── img_002.jpg
  │       └── ...
  └── val/
      ├── labels.txt
      └── images/
```

**Format of `labels.txt`:**

```text
img_001.png    Hello World
img_002.jpg    This is a test
```

*Note: Images must be in an `images/` subdirectory relative to the `labels.txt` file.*

### Step 2: Train the Recognition Model

You can train using CLI arguments or a configuration file.

**Option A: Using Configuration File (Recommended)**

1. **Generate default config**:
   ```bash
   kiri-ocr init-config -o config.json
   ```
2. **Edit `config.json`** to adjust hyperparameters (epochs, batch size, etc.).
3. **Start training**:
   ```bash
   kiri-ocr train --config config.json
   ```

**Option B: Using CLI Arguments**

```bash
kiri-ocr train \
    --train-labels data/train/labels.txt \
    --val-labels data/val/labels.txt \
    --epochs 100 \
    --batch-size 32 \
    --device cuda
```

**Option C: Training with Hugging Face Dataset**

You can train directly using a dataset from Hugging Face Hub. The dataset should contain `image` and `text` columns.

```bash
kiri-ocr train \
    --hf-dataset mrrtmob/km_en_image_line \
    --epochs 50 \
    --batch-size 32
```

**Advanced HF Options:**

* `--hf-train-split`: Specify training split name (default: "train").
* `--hf-val-split`: Specify validation split name. If not provided, it tries "validation", "val", "test", or automatically splits the training set.
* `--hf-val-percent`: Percentage of training data to use for validation if no validation split is found (default: 0.1 for 10%).
* `--hf-image-col`: Column name for images (default: "image").
* `--hf-text-col`: Column name for text labels (default: "text").
* `--hf-subset`: Dataset configuration/subset name (optional).
* `--hf-streaming`: Stream the dataset instead of downloading it fully.

To use a specific subset/config (if the dataset has multiple):

```bash
kiri-ocr train \
    --hf-dataset mrrtmob/km_en_image_line \
    --hf-subset default \
    ...
```

### Step 3: Train the Detector (Optional)

Kiri OCR uses a CRAFT-based detector (Character Region Awareness for Text Detection) to find text regions. You can train this on your own data if needed.

1. **Generate Detector Dataset**:

   Create a synthetic dataset for training the CRAFT detector. This command generates images with character-level ground truth maps (region and affinity).

   ```bash
   kiri-ocr generate-detector \
       --text-file data/textlines.txt \
       --fonts-dir fonts \
       --output detector_dataset \
       --num-train 1000 \
       --num-val 200
   ```

   **Arguments:**

   * `--text-file`: Path to source text file (one line per text sample).
   * `--fonts-dir`: Directory containing `.ttf` font files.
   * `--output`: Output directory for the dataset (default: `detector_dataset`).
   * `--num-train`: Number of training images to generate.
   * `--num-val`: Number of validation images to generate.
   * `--no-augment`: Disable random augmentations.

   This creates a `detector_dataset/` directory containing images and `.npy` label maps.

2. **Train Detector**:

   ```bash
   kiri-ocr train-detector \
       --epochs 50 \
       --batch-size 8 \
       --name my_craft_model
   ```

   **Arguments:**
   * `--epochs`: Number of training epochs.
   * `--batch-size`: Batch size (adjust based on your GPU memory).
   * `--name`: Name for the training run (models saved in `runs/detect/{name}`).

3. **Use Custom Detector**:
   The system will save the best model to `runs/detect/{name}/weights/best.pth`. You can also specify it in Python:

   ```python
   from kiri_ocr.detector import TextDetector
   detector = TextDetector(method='craft', model_path='runs/detect/my_craft_model/weights/best.pth')
   ```

### Fine-Tuning Recognition

To fine-tune an existing recognition model on new data:

```bash
kiri-ocr train \
    --config config.yaml \
    --from-model models/model.kiri
```

This loads the weights from `models/model.kiri` before starting training. Useful for domain adaptation or adding languages.

The trained model will be saved to `models/model.kiri` (or specified `output_dir`).

### ☕ Support

If you find this project useful, you can support me here:

- [Buy Me a Coffee](https://buymeacoffee.com/tmob)
- [ABA Payway](https://link.payway.com.kh/ABAPAYfd4073965)

## ⚖️ License

[Apache License 2.0](https://github.com/mrrtmob/kiri-ocr/blob/main/LICENSE).
