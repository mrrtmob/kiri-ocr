import numpy as np
import json
import argparse
import sys
from pathlib import Path
import yaml

from .core import OCR
from .renderer import DocumentRenderer
from .training import train_command
from .generator import generate_command

DEFAULT_TRAIN_CONFIG = {
    "height": 32,
    "batch_size": 32,
    "epochs": 2,
    "hidden_size": 256,
    "device": "cuda",
    "output_dir": "models",
    "train_labels": "data/train/labels.txt",
    "val_labels": "data/val/labels.txt",
    "lr": 0.001,
    "weight_decay": 0.0001
}

def init_config(args):
    path = args.output
    # Ensure .yaml extension if default or user didn't specify
    if path == 'config.yaml' and not path.endswith('.yaml') and not path.endswith('.yml'):
         pass # user provided custom name
    
    with open(path, 'w') as f:
        yaml.dump(DEFAULT_TRAIN_CONFIG, f, default_flow_style=False)
    print(f"✓ Created default config at {path}")

def run_inference(args):
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("  📄 Kiri OCR System")
    print("="*70)
    
    try:
        # Initialize OCR
        ocr = OCR(
            model_path=args.model,
            charset_path=args.charset,
            language=args.language,
            padding=args.padding,
            device=args.device,
            verbose=True
        )
        
        # Process document & Extract text
        full_text, results = ocr.extract_text(args.image, mode=args.mode, verbose=True)
        
        # Save text
        text_output = output_dir / 'extracted_text.txt'
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"\n✓ Text saved to {text_output}")
        
        # Save JSON
        json_output = output_dir / 'ocr_results.json'
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON saved to {json_output}")
        
        # Render results
        if not args.no_render:
            renderer = DocumentRenderer()
            
            # Boxes only
            renderer.draw_boxes(
                args.image, 
                results, 
                output_path=str(output_dir / 'boxes.png')
            )
            
            # Boxes with text
            renderer.draw_results(
                args.image,
                results,
                output_path=str(output_dir / 'ocr_result.png')
            )
            
            # HTML report
            renderer.create_report(
                args.image,
                results,
                output_path=str(output_dir / 'report.html')
            )
        
        print("\n" + "="*70)
        print("  ✅ Processing Complete!")
        print("="*70)
        print(f"  Regions detected: {len(results)}")
        if results:
            print(f"  Average confidence: {np.mean([r['confidence'] for r in results])*100:.2f}%")
        print(f"  Output directory: {output_dir}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Suggest model path if it seems to be missing
        if "No such file" in str(e) and "model" in str(e):
             print("\nTip: Make sure you have trained a model or specified the correct path with --model")
             print("     Run: kiri-ocr train ...  to train a model first.")

def merge_config(args, defaults):
    """Merge defaults < config file < args"""
    # Start with defaults
    config = defaults.copy()
    
    # Update with config file if provided
    if hasattr(args, 'config') and args.config:
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith('.json'):
                    file_config = json.load(f)
                else:
                    file_config = yaml.safe_load(f)
                
                if file_config:
                    config.update(file_config)
            print(f"Loaded config from {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
            
    # Update with explicit args (non-None)
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
            
    # Update args object
    for key, value in config.items():
        setattr(args, key, value)
        
    return args

def main():
    parser = argparse.ArgumentParser(description='Kiri OCR - Command Line Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # === PREDICT / RUN ===
    predict_parser = subparsers.add_parser('predict', help='Run OCR on an image')
    predict_parser.add_argument('image', help='Path to document image')
    predict_parser.add_argument('--mode', choices=['lines', 'words'], default='lines',
                       help='Detection mode (default: lines)')
    predict_parser.add_argument('--model', default='models/model.kiri',
                       help='Path to model file')
    predict_parser.add_argument('--charset', default='models/charset_lite.txt',
                       help='Path to charset file')
    predict_parser.add_argument('--language', choices=['english', 'khmer', 'mixed'], 
                       default='mixed', help='Language mode')
    predict_parser.add_argument('--padding', type=int, default=10,
                       help='Padding around detected boxes in pixels (default: 10)')
    predict_parser.add_argument('--output', '-o', default='output',
                       help='Output directory')
    predict_parser.add_argument('--no-render', action='store_true',
                       help='Skip rendering (text only)')
    predict_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use')
    
    # === TRAIN ===
    train_parser = subparsers.add_parser('train', help='Train the OCR model')
    train_parser.add_argument('--config', help='Path to config file (YAML/JSON)')
    train_parser.add_argument('--train-labels', help='Path to training labels file')
    train_parser.add_argument('--val-labels', help='Path to validation labels file')
    train_parser.add_argument('--hf-dataset', help='HuggingFace dataset ID (e.g. mrrtmob/km_en_image_line)')
    train_parser.add_argument('--hf-subset', help='HuggingFace dataset subset/config name')
    train_parser.add_argument('--hf-train-split', default='train', help='Train split name (default: train)')
    train_parser.add_argument('--hf-val-split', help='Validation split name')
    train_parser.add_argument('--hf-image-col', default='image', help='Image column name')
    train_parser.add_argument('--hf-text-col', default='text', help='Text column name')
    train_parser.add_argument('--hf-val-percent', type=float, default=0.1, help='Val % if no split found (default: 0.1)')
    train_parser.add_argument('--output-dir', help='Directory to save model')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--height', type=int, help='Image height')
    train_parser.add_argument('--hidden-size', type=int, help='Hidden size')
    train_parser.add_argument('--device', help='Device (cuda/cpu)')
    train_parser.add_argument('--from-model', help='Path to pretrained model for fine-tuning')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, help='Weight decay')
    
    # === GENERATE ===
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--train-file', '-t', required=True,
                       help='Training text file (one line per sample)')
    gen_parser.add_argument('--val-file', '-v', default=None,
                       help='Validation text file (optional)')
    gen_parser.add_argument('--output', '-o', default='data',
                       help='Output directory')
    gen_parser.add_argument('--language', '-l', choices=['english', 'khmer', 'mixed'],
                       default='mixed', help='Language mode')
    gen_parser.add_argument('--augment', '-a', type=int, default=1,
                       help='Augmentation factor for training')
    gen_parser.add_argument('--val-augment', type=int, default=1,
                       help='Augmentation factor for validation')
    gen_parser.add_argument('--height', type=int, default=32,
                       help='Image height')
    gen_parser.add_argument('--width', type=int, default=512,
                       help='Image width')
    gen_parser.add_argument('--fonts-dir', default='fonts',
                       help='Directory containing .ttf font files')
    gen_parser.add_argument('--font-mode', choices=['random', 'all'], default='random',
                       help='Font selection mode: random (default) or all (iterate all fonts per line)')
    gen_parser.add_argument('--random-augment', action='store_true',
                       help='Apply random augmentations (noise, rotation) even if augmentation factor is 1')

    # === INIT CONFIG ===
    init_parser = subparsers.add_parser('init-config', help='Create default config file')
    init_parser.add_argument('--output', '-o', default='config.yaml', help='Output file')

    # Backward compatibility logic
    if len(sys.argv) > 1 and sys.argv[1] not in ['predict', 'train', 'generate', 'init-config', '-h', '--help']:
        sys.argv.insert(1, 'predict')
        
    args = parser.parse_args()
    
    if args.command == 'predict':
        run_inference(args)
    elif args.command == 'train':
        # Merge config for training
        args = merge_config(args, DEFAULT_TRAIN_CONFIG)
        # Check required fields
        if not args.train_labels and not args.hf_dataset:
            print("❌ Error: --train-labels or --hf-dataset is required")
            sys.exit(1)
            
        train_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'init-config':
        init_config(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
