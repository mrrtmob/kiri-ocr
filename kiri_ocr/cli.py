# kiri_ocr/cli.py
import json
import argparse
import sys
import time
from pathlib import Path
import yaml

# Updated default configs for both architectures
DEFAULT_TRAIN_CONFIG = {
    "arch": "crnn",
    "height": 32,
    "width": 512,
    "batch_size": 32,
    "epochs": 100,
    "hidden_size": 256,
    "device": "cuda",
    "output_dir": "models",
    "train_labels": "data/train/labels.txt",
    "val_labels": "data/val/labels.txt",
    "vocab": None,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "save_steps": 0,
    "resume": False,
    "from_model": None,
}

# Transformer-specific defaults (merged when arch=transformer)
TRANSFORMER_DEFAULTS = {
    "height": 48,
    "width": 640,
    "lr": 0.0003,
    "weight_decay": 0.01,
    "ctc_weight": 0.5,
    "dec_weight": 0.5,
    "save_steps": 1000,
}


def init_config(args):
    path = args.output
    config = DEFAULT_TRAIN_CONFIG.copy()

    # Add transformer section as comment
    config_with_comments = f"""# Kiri OCR Training Configuration
# Architecture: 'crnn' (fast, lightweight) or 'transformer' (accurate, slower)

arch: crnn

# Image dimensions (CRNN: 32x512, Transformer: 48x640)
height: 32
width: 512

# Training parameters
batch_size: 32
epochs: 100
lr: 0.001
weight_decay: 0.0001

# Device
device: cuda

# Paths
output_dir: models
train_labels: data/train/labels.txt
val_labels: data/val/labels.txt

# CRNN specific
hidden_size: 256

# Transformer specific (uncomment if using arch: transformer)
# vocab: vocab_char.json
# ctc_weight: 0.5
# dec_weight: 0.5
# save_steps: 1000

# Resume training
resume: false
# from_model: path/to/pretrained.pt

# HuggingFace dataset (alternative to local labels)
# hf_dataset: username/dataset_name
# hf_train_split: train
# hf_val_split: validation
# hf_image_col: image
# hf_text_col: text
"""

    with open(path, "w") as f:
        f.write(config_with_comments)
    print(f"✓ Created default config at {path}")


def run_inference(args):
    import numpy as np
    from .core import OCR
    from .renderer import DocumentRenderer

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    if args.verbose:
        print("\n" + "=" * 70)
        print("  📄 Kiri OCR System")
        print("=" * 70)

    try:
        ocr = OCR(
            model_path=args.model,
            charset_path=args.charset,
            language=args.language,
            padding=args.padding,
            device=args.device,
            verbose=args.verbose,
        )

        if not args.verbose:
            print(f"Processing {args.image}...")

        full_text, results = ocr.extract_text(
            args.image, mode=args.mode, verbose=args.verbose
        )

        text_output = output_dir / "extracted_text.txt"
        with open(text_output, "w", encoding="utf-8") as f:
            f.write(full_text)
        if args.verbose:
            print(f"\n✓ Text saved to {text_output}")

        json_output = output_dir / "ocr_results.json"
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if args.verbose:
            print(f"✓ JSON saved to {json_output}")

        if not args.no_render:
            renderer = DocumentRenderer()
            renderer.draw_boxes(
                args.image, results, output_path=str(output_dir / "boxes.png")
            )
            renderer.draw_results(
                args.image, results, output_path=str(output_dir / "ocr_result.png")
            )
            renderer.create_report(
                args.image, results, output_path=str(output_dir / "report.html")
            )

        if args.verbose:
            print("\n" + "=" * 70)
            print("  ✅ Processing Complete!")
            print("=" * 70)
            print(f"  Regions detected: {len(results)}")
            if results:
                print(
                    f"  Average confidence: {np.mean([r['confidence'] for r in results])*100:.2f}%"
                )
            print(f"  Output directory: {output_dir}")
            print("=" * 70 + "\n")
        else:
            if results:
                for res in results:
                    print(res["text"])
            print(f"\n✓ Saved results to {output_dir}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


def merge_config(args, defaults):
    """Merge defaults < config file < CLI args"""
    config = defaults.copy()

    # Load config file if provided
    if hasattr(args, "config") and args.config:
        try:
            with open(args.config, "r") as f:
                if args.config.endswith(".json"):
                    file_config = json.load(f)
                else:
                    file_config = yaml.safe_load(f)

                if file_config:
                    config.update(file_config)
            print(f"📁 Loaded config from {args.config}")
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            sys.exit(1)

    # CLI args override everything (only if not None)
    for key, value in vars(args).items():
        if value is not None:
            # Convert dashes to underscores
            config_key = key.replace("-", "_")
            if config_key in config or key in config:
                config[config_key] = value
            else:
                config[key] = value

    # Apply all config values to args
    for key, value in config.items():
        setattr(args, key, value)

    return args


def print_banner(version="0.0.0"):
    banner = f"""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     ██╗  ██╗██╗██████╗ ██╗      ██████╗  ██████╗██████╗        ║
║     ██║ ██╔╝██║██╔══██╗██║     ██╔═══██╗██╔════╝██╔══██╗       ║
║     █████╔╝ ██║██████╔╝██║     ██║   ██║██║     ██████╔╝       ║
║     ██╔═██╗ ██║██╔══██╗██║     ██║   ██║██║     ██╔══██╗       ║
║     ██║  ██╗██║██║  ██║██║     ╚██████╔╝╚██████╗██║  ██║       ║
║     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝      ╚═════╝  ╚═════╝╚═╝  ╚═╝       ║
║                                                                ║
║            Khmer & English OCR System                          ║
║                   Version: {version:^10}                          ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    start_setup = time.time()
    try:
        from . import __version__
    except ImportError:
        __version__ = "0.0.0"

    if "--version" in sys.argv:
        print_banner(__version__)
        sys.exit(0)

    show_banner = len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv
    if show_banner:
        print_banner(__version__)

    parser = argparse.ArgumentParser(
        description="Kiri OCR - Khmer & English OCR System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========== PREDICT ==========
    base_path = Path(__file__).parent
    default_model = "mrrtmob/kiri-ocr"
    default_charset = base_path / "models" / "charset_lite.txt"

    predict_parser = subparsers.add_parser("predict", help="🔍 Run OCR on an image")
    predict_parser.add_argument("image", help="Path to document image")
    predict_parser.add_argument("--mode", choices=["lines", "words"], default="lines")
    predict_parser.add_argument("--model", default=str(default_model))
    predict_parser.add_argument("--charset", default=str(default_charset))
    predict_parser.add_argument(
        "--language", choices=["english", "khmer", "mixed"], default="mixed"
    )
    predict_parser.add_argument("--padding", type=int, default=10)
    predict_parser.add_argument("--output", "-o", default="output")
    predict_parser.add_argument("--no-render", action="store_true")
    predict_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    predict_parser.add_argument("--verbose", "-v", action="store_true")

    # ========== TRAIN ==========
    train_parser = subparsers.add_parser("train", help="🎓 Train the OCR model")

    # Architecture
    train_parser.add_argument(
        "--arch",
        choices=["crnn", "transformer"],
        default=None,
        help="Model architecture (default: from config or crnn)",
    )
    train_parser.add_argument("--config", help="Path to config file (YAML or JSON)")

    # Data sources
    train_parser.add_argument("--train-labels", help="Path to training labels.txt")
    train_parser.add_argument("--val-labels", help="Path to validation labels.txt")

    # HuggingFace dataset
    train_parser.add_argument("--hf-dataset", help="HuggingFace dataset ID")
    train_parser.add_argument("--hf-subset", help="Dataset subset/config")
    train_parser.add_argument("--hf-train-split", default="train")
    train_parser.add_argument("--hf-val-split", default=None)
    train_parser.add_argument("--hf-streaming", action="store_true")
    train_parser.add_argument("--hf-image-col", default="image")
    train_parser.add_argument("--hf-text-col", default="text")
    train_parser.add_argument("--hf-val-percent", type=float, default=0.1)

    # Image dimensions
    train_parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Image height (CRNN: 32, Transformer: 48)",
    )
    train_parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Image width (CRNN: 512, Transformer: 640)",
    )

    # Training hyperparameters
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=None)

    # CRNN specific
    train_parser.add_argument("--hidden-size", type=int, default=None)

    # Transformer specific
    train_parser.add_argument("--vocab", help="Path to vocab_char.json")
    train_parser.add_argument(
        "--ctc-weight",
        type=float,
        default=None,
        help="CTC loss weight (transformer only)",
    )
    train_parser.add_argument(
        "--dec-weight",
        type=float,
        default=None,
        help="Decoder loss weight (transformer only)",
    )

    # Checkpointing
    train_parser.add_argument("--output-dir", help="Output directory")
    train_parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (0=epoch only)",
    )
    train_parser.add_argument(
        "--resume", action="store_true", help="Resume from latest.pt"
    )
    train_parser.add_argument("--from-model", help="Initialize from pretrained model")

    # Device
    train_parser.add_argument("--device", choices=["cpu", "cuda"], default=None)

    # ========== GENERATE ==========
    gen_parser = subparsers.add_parser(
        "generate", help="🎨 Generate synthetic training data"
    )
    gen_parser.add_argument("--train-file", "-t", required=True)
    gen_parser.add_argument("--val-file", "-v", default=None)
    gen_parser.add_argument("--output", "-o", default="data")
    gen_parser.add_argument(
        "--language", "-l", choices=["english", "khmer", "mixed"], default="mixed"
    )
    gen_parser.add_argument("--augment", "-a", type=int, default=1)
    gen_parser.add_argument("--val-augment", type=int, default=1)
    gen_parser.add_argument("--height", type=int, default=32)
    gen_parser.add_argument("--width", type=int, default=512)
    gen_parser.add_argument("--fonts-dir", default="fonts")
    gen_parser.add_argument("--font-mode", choices=["random", "all"], default="random")
    gen_parser.add_argument("--random-augment", action="store_true")

    # ========== DETECTOR TOOLS ==========
    gen_det_parser = subparsers.add_parser(
        "generate-detector", help="🖼️ Generate detector data"
    )
    gen_det_parser.add_argument("--text-file", required=True)
    gen_det_parser.add_argument("--fonts-dir", default="fonts")
    gen_det_parser.add_argument("--font", help="Specific font")
    gen_det_parser.add_argument("--output", default="detector_dataset")
    gen_det_parser.add_argument("--num-train", type=int, default=800)
    gen_det_parser.add_argument("--num-val", type=int, default=200)
    gen_det_parser.add_argument("--min-lines", type=int, default=15)
    gen_det_parser.add_argument("--max-lines", type=int, default=50)
    gen_det_parser.add_argument("--image-height", type=int, default=512)
    gen_det_parser.add_argument("--no-augment", action="store_true")
    gen_det_parser.add_argument("--workers", type=int, default=1)

    train_det_parser = subparsers.add_parser(
        "train-detector", help="🎯 Train text detector"
    )
    train_det_parser.add_argument("--data-yaml", default="detector_dataset/data.yaml")
    train_det_parser.add_argument(
        "--model-size", choices=["n", "s", "m", "l", "x"], default="n"
    )
    train_det_parser.add_argument("--epochs", type=int, default=100)
    train_det_parser.add_argument("--batch-size", type=int, default=16)
    train_det_parser.add_argument("--image-size", type=int, default=640)
    train_det_parser.add_argument("--name", default="khmer_text_detector")

    # ========== CONFIG ==========
    init_parser = subparsers.add_parser("init-config", help="⚙️ Create config file")
    init_parser.add_argument("--output", "-o", default="config.yaml")

    # Default to predict if first arg looks like a file
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        commands = [
            "predict",
            "train",
            "generate",
            "init-config",
            "generate-detector",
            "train-detector",
            "-h",
            "--help",
            "--version",
        ]
        if first_arg not in commands and not first_arg.startswith("-"):
            sys.argv.insert(1, "predict")

    args = parser.parse_args()

    if hasattr(args, "help") and args.help:
        parser.print_help()
        sys.exit(0)

    # ========== COMMAND ROUTING ==========
    if args.command == "predict":
        run_inference(args)

    elif args.command == "train":
        # Merge config
        args = merge_config(args, DEFAULT_TRAIN_CONFIG)

        # Validate data source
        if not getattr(args, "train_labels", None) and not getattr(
            args, "hf_dataset", None
        ):
            print("❌ Error: --train-labels or --hf-dataset is required")
            sys.exit(1)

        # Get architecture
        arch = getattr(args, "arch", "crnn")

        if arch == "transformer":
            print("\n" + "=" * 60)
            print("  🚀 Transformer (Infinity) Training")
            print("=" * 60)

            # Apply transformer defaults for unset values
            for key, value in TRANSFORMER_DEFAULTS.items():
                if getattr(args, key, None) is None:
                    setattr(args, key, value)

            print(f"\n📐 Image size: {args.height}x{args.width}")
            print(f"⚖️  Loss weights: CTC={args.ctc_weight}, Decoder={args.dec_weight}")
            print(f"📊 Batch size: {args.batch_size}")
            print(f"🎯 Learning rate: {args.lr}")
            print(f"💾 Save steps: {args.save_steps}")

            try:
                from .training_transformer import train_command as train_transformer_cmd

                train_transformer_cmd(args)
            except ImportError as e:
                print(f"❌ Error: Could not import training_transformer: {e}")
                print("   Make sure training_transformer.py exists in kiri_ocr/")
                sys.exit(1)
        else:
            # CRNN Training
            print("\n" + "=" * 60)
            print("  🚀 CRNN (Lightweight) Training")
            print("=" * 60)

            # Apply CRNN defaults
            if args.height is None:
                args.height = 32
            if args.width is None:
                args.width = 512

            from .training import train_command

            train_command(args)

    elif args.command == "generate":
        from .generator import generate_command

        generate_command(args)

    elif args.command == "generate-detector":
        from .detector.craft.dataset import generate_detector_dataset_command

        generate_detector_dataset_command(args)

    elif args.command == "train-detector":
        from .detector.craft.training import train_detector_command

        train_detector_command(args)

    elif args.command == "init-config":
        init_config(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
